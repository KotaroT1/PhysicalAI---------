"""ローカル LLM (Qwen2.5-Instruct) による日本語指示 → ピック&プレース計画への変換。

小型 LLM でも確実に動かすための工夫:
  1. 「手順の列」を自由に書かせるのではなく、「全物体それぞれの行き先」を埋めさせる
     (スロット埋め形式)。条件に合う物体の取りこぼしが減る。
  2. 既定の choice モードでは、各物体について行き先の候補 (stay / 容器 / 他の物体) を
     LLM に採点させて最尤のものを選ぶ。シーンに無い名前や壊れた JSON は原理的に出ない。
  3. generate モードでは出力をシーンと照合して検証し、不正なら誤りをフィードバックして
     再生成、それでも駄目ならルールベース解析器にフォールバックする。
  4. 実行順序 (積み上げの依存関係) は LLM に任せず、プログラムで決める。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# 動作確認したコミット (再現性のため固定)
REVISIONS = {"Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
             "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}

SYSTEM_PROMPT = """You are the task planner of a robot arm that tidies up a table.
Decide the destination of EVERY object in OBJECTS according to the user's Japanese instruction.

A destination is one of:
- a name from PLACES,
- a name from OBJECTS (= stack it on top of that object),
- "stay" (do not move it).

Rules:
- Words: ブロック/キューブ/積み木 = cube, ボール/玉/球 = ball, 缶/円柱 = cylinder, 箱/木箱 = wooden box, 皿/お皿 = white plate.
- 丸いもの (round things) = balls and cylinders. 四角いもの/角ばったもの = cubes.
- 「全部」「すべて」 = every object that matches the condition.
- 「片付けて」 without a condition = every object. If no place is given, use the wooden box.
- Objects that the instruction does not mention must "stay".
- Use the position in parentheses for left/right/front/back.
{think_rule}- "reply": one short polite Japanese sentence (〜します。) the robot says before acting.
Output ONLY one JSON object: {format}"""

THINK_RULE = '- "think": one short English sentence: which objects match and where they go.\n'
FORMAT_THINK = '{"think": "...", "moves": {"<object>": "<destination>", ...}, "reply": "..."}'
FORMAT_PLAIN = '{"moves": {"<object>": "<destination>", ...}, "reply": "..."}'

# (user, think, moves, reply)
FEW_SHOT = [
    ("OBJECTS: purple cube (left), white cube (right), pink ball (center)\n"
     "PLACES: wooden box, white plate\n"
     "INSTRUCTION: ピンクのボールをお皿にのせて",
     "pink ball goes to the white plate.",
     {"purple cube": "stay", "white cube": "stay", "pink ball": "white plate"},
     "ピンクのボールをお皿にのせます。"),
    ("OBJECTS: white cylinder (back left), black cube (front), pink ball (right), gray cube (center)\n"
     "PLACES: wooden box, white plate\n"
     "INSTRUCTION: 丸いものは全部お皿に移して",
     "round things are white cylinder and pink ball; they go to the white plate.",
     {"white cylinder": "white plate", "black cube": "stay", "pink ball": "white plate", "gray cube": "stay"},
     "丸いもの2個をお皿に移します。"),
    ("OBJECTS: black cube (left), gray cube (center), brown ball (right)\n"
     "PLACES: wooden box, white plate\n"
     "INSTRUCTION: 灰色のブロックを黒いブロックの上に積んで",
     "gray cube is stacked on the black cube.",
     {"black cube": "stay", "gray cube": "black cube", "brown ball": "stay"},
     "灰色のブロックを黒いブロックの上に積みます。"),
]


def _render(think: str, moves: dict, reply: str, with_think: bool) -> str:
    d = ({"think": think} if with_think else {}) | {"moves": moves, "reply": reply}
    return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------- 分解方式 (decompose)
# 段階1: 指示を「何を (英語の説明句) → どこへ」の規則に言い換える
PARSE_SYSTEM = """You convert a Japanese instruction for a table-tidying robot into JSON rules.
Output ONLY: {"rules": [{"targets": ["..."], "dest": "..."}], "reply": "..."}
- "targets": a list of simple English noun phrases, one per kind of object to move.
  Use "the ..." for exactly one object (e.g. "the red cube"), plural nouns for every object of a kind (e.g. "cubes").
  Never join two kinds with "and"; list them separately (e.g. ["balls", "cylinders"]).
- "dest": "wooden box", "white plate", or "on the ..." to stack on another object.
- Use only these object words: cube, ball, cylinder (ブロック/キューブ/積み木 = cube, ボール/玉/球 = ball, 缶/円柱 = cylinder).
  箱/木箱 = wooden box, 皿/お皿 = white plate, 丸いもの = balls and cylinders, 四角いもの/角ばったもの = cubes.
- If no destination is given (e.g. 片付けて), use "wooden box".
- "reply": one short polite Japanese sentence (〜します。) the robot says before acting."""

PARSE_SHOTS = [
    ("四角いものを全部お皿に移して",
     {"rules": [{"targets": ["cubes"], "dest": "white plate"}], "reply": "四角いものをすべてお皿に移します。"}),
    ("紫のボールと白と黒の積み木を箱へ",
     {"rules": [{"targets": ["the purple ball", "the white cube", "the black cube"], "dest": "wooden box"}],
      "reply": "紫のボールと白と黒のブロックを木箱に入れます。"}),
    ("一番手前のものは箱に入れて、白いブロックは黒いブロックの上にのせて",
     {"rules": [{"targets": ["the frontmost object"], "dest": "wooden box"},
                {"targets": ["the white cube"], "dest": "on the black cube"}],
      "reply": "一番手前のものを木箱に入れ、白いブロックを黒いブロックの上にのせます。"}),
    ("四角いものと円柱は全部箱にしまって",
     {"rules": [{"targets": ["cubes", "cylinders"], "dest": "wooden box"}],
      "reply": "四角いものと円柱をすべて木箱にしまいます。"}),
    ("テーブルの上のものを全部お皿にまとめて",
     {"rules": [{"targets": ["all objects"], "dest": "white plate"}], "reply": "テーブルの上のものをすべてお皿にまとめます。"}),
    ("丸いものだけ箱に入れて",
     {"rules": [{"targets": ["balls", "cylinders"], "dest": "wooden box"}], "reply": "丸いものを木箱に入れます。"}),
]

# 段階2: 各物体が説明句に当てはまるかを yes / no で判定する
# 位置を含まない説明句は物体名だけで判定する (位置の一覧を見せると 3B モデルが例文の位置表現に引きずられたため)
MATCH_SYSTEM = "You check whether an object matches a description. Answer only yes or no."
NAME_SHOTS = [
    ("balls", "white ball", "yes"), ("balls", "purple cube", "no"), ("cylinders", "gray cylinder", "yes"),
    ("the pink ball", "white cube", "no"), ("the pink ball", "pink ball", "yes"), ("the pink ball", "black ball", "no"),
    ("the white cylinder", "white cylinder", "yes"), ("cubes", "gray cube", "yes"), ("cubes", "white cylinder", "no"),
    ("all objects", "pink ball", "yes"),
]
# 同義語はコード側で正規化する (block / cube の取り違えなどを防ぐ)
SYNONYMS = [(r"\bblocks\b", "cubes"), (r"\bblock\b", "cube"), (r"\bcans\b", "cylinders"), (r"\bcan\b", "cylinder"),
            (r"\bspheres\b", "balls"), (r"\bsphere\b", "ball")]
SPATIAL_WORDS = re.compile(r"\b(left|right|front|back|middle|center|centre|near|nearest|far|farthest|closest|"
                           r"leftmost|rightmost|frontmost|backmost|top|bottom)\b", re.I)
# 位置を含む説明句は、物体の一覧 (位置つき) を見せて判定する
MATCH_POS_SYSTEM = "You check whether an object on a table matches a description. Answer only yes or no."
MATCH_SHOTS = [
    ("purple cube (left), white ball (right), gray cylinder (back center)", "round objects",
     [("white ball", "yes"), ("purple cube", "no"), ("gray cylinder", "yes")]),
    ("purple cube (left), white ball (right), gray cylinder (back center)", "the rightmost object",
     [("purple cube", "no"), ("white ball", "yes")]),
    ("black cube (front), pink ball (center), white cube (back left)", "the black cube and the pink ball",
     [("white cube", "no"), ("pink ball", "yes"), ("black cube", "yes")]),
    ("white cylinder (left), black ball (right)", "the white can",
     [("black ball", "no"), ("white cylinder", "yes")]),
]


@dataclass
class Plan:
    steps: list  # [{"pick": str, "place": str}] (実行順)
    reply: str
    raw: str = ""
    source: str = "llm"  # "llm" | "llm-retry" | "rule"
    seconds: float = 0.0
    errors: list = field(default_factory=list)
    think: str = ""
    status: str = "ok"  # "ok" | "empty" (該当なし) | "invalid" (計画不能)


def validate(obj, objects: list[str], places: list[str]) -> list[str]:
    if not isinstance(obj, dict) or not isinstance(obj.get("moves"), dict):
        return ['output must be {"think": ..., "moves": {...}, "reply": ...}']
    moves = obj["moves"]
    errs = []
    for k, v in moves.items():
        if k not in objects:
            errs.append(f'"{k}" is not in OBJECTS')
        elif v != "stay" and v not in places and v not in objects:
            errs.append(f'destination "{v}" of "{k}" is not "stay", a PLACE or an OBJECT')
        elif v == k:
            errs.append(f'"{k}" cannot be stacked on itself')
    missing = [o for o in objects if o not in moves]
    if missing:
        errs.append("missing objects: " + ", ".join(missing))
    stack_targets = [v for v in moves.values() if v in objects]
    if len(stack_targets) != len(set(stack_targets)):
        errs.append("two objects cannot be stacked on the same object")
    if not errs and all(v == "stay" for v in moves.values()):
        errs.append("nothing moves; at least one object must move")
    if not errs and _order(moves) is None:
        errs.append("stacking has a cycle")
    return errs


def _order(moves: dict):
    """積み上げの依存関係を考慮した実行順 (土台を先に動かす)。循環があれば None。"""
    todo = {k: v for k, v in moves.items() if v != "stay"}
    order, done = [], set()
    while todo:
        ready = [k for k, v in todo.items() if v not in todo]  # 行き先の物体がもう動かない
        if not ready:
            return None
        for k in ready:
            order.append({"pick": k, "place": todo.pop(k)})
            done.add(k)
    return order


SHAPE_NOUNS = {"cube": "cube", "cubes": "cube", "ball": "ball", "balls": "ball",
               "cylinder": "cylinder", "cylinders": "cylinder"}
ALL_WORDS = {"objects", "things", "items", "everything", "object", "thing"}
COLOR_WORDS = {"red", "green", "blue", "yellow", "orange", "purple", "white", "black", "pink", "gray", "grey",
               "brown", "violet", "cyan", "magenta"}


def _lexical_match(target: str, objects: list[str]):
    """説明句を物体ラベル (「色 形」) と字面で照合する。決まれば該当物体のリスト (空 = 該当なし)、
    名前だけでは決まらない (位置・性質など) なら None を返し、LLM の判定に回す。"""
    t = target.lower().strip().rstrip(".")
    t = re.sub(r"\s+(on|from) the (table|desk)$", "", t)
    while True:
        t2 = re.sub(r"^(all|every|each|the|a|an|of)\s+", "", t)
        if t2 == t:
            break
        t = t2
    words = t.split()
    if not words or SPATIAL_WORDS.search(t):
        return None
    if t in ALL_WORDS:
        return list(objects)
    if len(words) == 1 and words[0] in SHAPE_NOUNS:
        return [o for o in objects if o.split()[-1] == SHAPE_NOUNS[words[0]]]
    colors = COLOR_WORDS | {o.split()[0] for o in objects if len(o.split()) == 2}
    if len(words) == 2 and words[1] in SHAPE_NOUNS and words[0] in colors:
        return [o for o in objects if o == f"{words[0]} {SHAPE_NOUNS[words[1]]}"]
    return None


def _normalize(phrase: str) -> str:
    out = phrase.strip()
    for pat, rp in SYNONYMS:
        out = re.sub(pat, rp, out, flags=re.I)
    return out


def _name_question(desc: str, obj: str) -> str:
    return f'Description: "{desc}"\nObject: "{obj}"\nDoes the object match the description?'


def _is_singular(target: str) -> bool:
    """「the red cube」のように 1 個を指す説明句か (複数形・and・all/every を含むなら複数)。"""
    t = target.strip().lower().rstrip(".")
    words = t.split()
    if not words or " and " in t or words[0] in ("all", "every", "each") or "all " in t:
        return False
    return words[0] in ("the", "a", "an") and not words[-1].endswith("s")


def _match_place(dest: str, places: list[str]):
    d = dest.lower()
    for p in places:
        if p in d or p.split()[-1] in d:
            return p
    return None


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class LLMPlanner:
    """mode="choice": 各物体の行き先候補をそれぞれ LLM に採点させ、最も尤度の高い候補を選ぶ (制約付き選択)。
                     形式エラーが原理的に起きず、シーンに無い名前も出ない。
       mode="generate": JSON を自由生成させ、検証 → 自己修正 → ルールにフォールバック。"""

    def __init__(self, model_id: str = MODEL_ID, mode: str = "decompose", think: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        rev = REVISIONS.get(model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
        # 3B は fp32 だと RAM 16GB に収まらないため bf16 で読み込む
        dtype = torch.bfloat16 if "3B" in model_id else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, revision=rev).eval()
        self.model_id = model_id
        self.mode = mode
        self.think = think

    @property
    def short_name(self) -> str:
        return self.model_id.split("/")[-1].replace("-Instruct", "")

    def _messages(self, instruction, objects, places, descriptions):
        obj_txt = ", ".join(f"{o} ({descriptions[o]})" if o in descriptions else o for o in objects)
        user = f"OBJECTS: {obj_txt}\nPLACES: {', '.join(places)}\nINSTRUCTION: {instruction}"
        system = SYSTEM_PROMPT.replace("{think_rule}", THINK_RULE if self.think else "").replace(
            "{format}", FORMAT_THINK if self.think else FORMAT_PLAIN)
        messages = [{"role": "system", "content": system}]
        for u, th, mv, rp in FEW_SHOT:
            messages += [{"role": "user", "content": u},
                         {"role": "assistant", "content": _render(th, mv, rp, self.think)}]
        messages.append({"role": "user", "content": user})
        return messages

    def plan(self, instruction: str, objects: list[str], places: list[str],
             descriptions: dict[str, str] | None = None) -> Plan:
        if self.mode == "decompose":
            return self._plan_decompose(instruction, objects, places, descriptions or {})
        messages = self._messages(instruction, objects, places, descriptions or {})
        if self.mode == "choice":
            return self._plan_choice(messages, objects, places)
        return self._plan_generate(messages, objects, places, instruction)

    # ------------------------------------------------------------------ 自由生成 + 検証
    def _generate(self, messages, max_new_tokens: int = 320) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        return self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def _plan_generate(self, messages, objects, places, instruction) -> Plan:
        t0 = time.time()
        all_errs = []
        raw = ""
        for attempt in range(2):
            raw = self._generate(messages)
            obj = _extract_json(raw)
            errs = validate(obj, objects, places) if obj is not None else ["output is not valid JSON"]
            if not errs:
                moves = {o: obj["moves"][o] for o in objects}  # シーン内の並び順に揃える
                return Plan(_order(moves), str(obj.get("reply", "")), raw,
                            "llm" if attempt == 0 else "llm-retry", time.time() - t0, all_errs,
                            str(obj.get("think", "")))
            all_errs += errs
            # 誤りをフィードバックして再生成 (self-correction)
            messages = messages + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": "Your answer has errors: " + "; ".join(errs)
                                    + ". Fix them and output only the corrected JSON."}]
        rule = rule_based_plan(instruction, objects, places)
        rule.raw, rule.seconds, rule.errors = raw, time.time() - t0, all_errs
        return rule

    # ------------------------------------------------------------------ 制約付き選択
    def _forward(self, ids: list[int], max_keep: int | None = None):
        """KV キャッシュを共通接頭辞まで巻き戻してから差分だけ推論し、新規部分のロジットを返す。"""
        L = 0
        for a, b in zip(self._ids, ids):
            if a != b:
                break
            L += 1
        L = min(L, len(ids) - 1, len(ids) if max_keep is None else max_keep)
        self._cache.crop(L)
        with self.torch.no_grad():
            out = self.model(input_ids=self.torch.tensor([ids[L:]]), past_key_values=self._cache, use_cache=True)
        self._ids = list(ids)
        return out.logits[0].float(), L

    def _greedy_until_quote(self, text: str, max_tokens: int = 60) -> str:
        ids = self.tok(text).input_ids
        logits, _ = self._forward(ids)
        gen = []
        for _ in range(max_tokens):
            t = int(logits[-1].argmax())
            if t == self.tok.eos_token_id:
                break
            gen.append(t)
            if '"' in self.tok.decode(gen):
                break
            logits, _ = self._forward(ids + gen)
        return self.tok.decode(gen).split('"')[0]

    def _score(self, text: str, cand: str, suffix: str) -> float:
        """text の直後に cand + suffix が続く対数尤度。suffix は実際の続き (例文と同じトークン区切りにする)。"""
        base = self.tok(text).input_ids
        full = self.tok(text + cand + suffix).input_ids
        # 候補の 1 トークン目を予測するロジットは text の最後のトークン位置にあるので、そこから再計算する
        logits, L = self._forward(full, max_keep=len(base) - 1)
        logp = self.torch.log_softmax(logits, -1)
        # logits[j] は full[L + j + 1] を予測する。text 以降 (候補 + 閉じ引用符) の対数尤度の合計
        return float(sum(logp[i - L - 1, full[i]] for i in range(len(base), len(full))))

    # ------------------------------------------------------------------ 分解方式
    def _plan_decompose(self, instruction, objects, places, descriptions) -> Plan:
        """段階1: 指示 → {何を, どこへ} の規則 (生成)。段階2: 各物体が「何を」に当てはまるか (yes/no の採点)。"""
        from transformers import DynamicCache

        t0 = time.time()
        # 段階1 -----------------------------------------------------------
        msgs = [{"role": "system", "content": PARSE_SYSTEM}]
        for ins, out in PARSE_SHOTS:
            msgs += [{"role": "user", "content": ins},
                     {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)}]
        msgs.append({"role": "user", "content": instruction})
        self._use_cache("parse")  # 固定の前置き (指示文より前) は前回の計算結果を再利用
        raw = self._greedy(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), 160)
        self._store_cache("parse")
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("rules"), list):
            rule = rule_based_plan(instruction, objects, places)
            rule.raw, rule.seconds, rule.errors = raw, time.time() - t0, ["parse failed"]
            return rule
        # 段階2 -----------------------------------------------------------
        self._use_cache("match")
        scene = ", ".join(f"{o} ({descriptions[o]})" if o in descriptions else o for o in objects)
        details, problems, lexical = [], [], []
        for r in parsed["rules"]:
            if not isinstance(r, dict):
                continue
            targets = r.get("targets", r.get("target", []))
            if isinstance(targets, str):
                targets = [targets]
            elif isinstance(targets, list):
                targets = [str(t) for t in targets if isinstance(t, (str, int, float)) and str(t).strip()]
            else:
                problems.append("rule without valid targets")
                targets = []
            targets = [_normalize(t) for t in targets]
            dest = _normalize(str(r.get("dest", "")))
            p_rule, chosen = {o: 0.0 for o in objects}, []
            for target in targets:
                lex = _lexical_match(target, objects)
                if lex is not None:  # 名前で決まる説明句はプログラムで照合 (確率は 1/0)
                    p_yes = {o: float(o in lex) for o in objects}
                    lexical.append(target)
                else:  # 位置や性質など、名前だけでは決まらない説明句は LLM が判定
                    p_yes = {o: self._p_yes(scene, target, o) for o in objects}
                if not p_yes:
                    continue
                if _is_singular(target):
                    best = max(p_yes, key=p_yes.get)
                    hit = [best] if p_yes[best] > 0.5 else []
                else:
                    hit = [o for o, pv in p_yes.items() if pv > 0.5]
                if not hit:
                    problems.append(f'no object matches "{target}"')
                chosen += [o for o in hit if o not in chosen]
                p_rule = {o: max(p_rule[o], p_yes[o]) for o in objects}
            if targets:
                details.append({"target": ", ".join(targets), "dest": dest, "p_yes": p_rule, "chosen": chosen,
                                "place": None, "specific": all(_is_singular(t) for t in targets)})
        # 複数の規則に当てはまった物体は、1 個を指す具体的な規則を優先し (「ブロックは箱へ、ただし左端のブロックは皿へ」)、
        # 同じ具体性なら yes 確率が最も高い規則に割り当てる (規則同士を競わせる)
        for o in objects:
            hits = [d for d in details if o in d["chosen"]]
            for d in hits:
                if d is not max(hits, key=lambda h: (h["specific"], h["p_yes"][o])):
                    d["chosen"].remove(o)
        moves = {o: "stay" for o in objects}
        for d in details:
            place = _match_place(d["dest"], places)
            if place is None and d["dest"].lower().startswith("on "):
                ref = d["dest"][3:].strip()
                lex = _lexical_match(ref, objects)
                cands = [o for o in objects if o not in d["chosen"]]
                p_ref = ({o: float(o in lex) for o in cands} if lex is not None
                         else {o: self._p_yes(scene, ref, o) for o in cands})
                best = max(p_ref, key=p_ref.get) if p_ref else None
                place = best if best is not None and p_ref[best] > 0.5 else None
            d["place"] = place
            if place is None:
                if d["chosen"]:
                    problems.append(f'destination "{d["dest"]}" not found')
                continue
            for o in d["chosen"]:
                if o != place:
                    moves[o] = place
        self._store_cache("match")
        for d in details:
            d["p_yes"] = {k: round(v, 2) for k, v in d["p_yes"].items()}
        # 計画の検証 (同じ土台に複数を積む・循環) は一括生成方式と同じ規則で行う
        errs = validate({"moves": moves}, objects, places) if any(v != "stay" for v in moves.values()) else []
        steps = _order(moves) if not errs else None
        status = "ok" if steps else ("invalid" if errs else "empty")
        info = json.dumps({"rules": parsed["rules"], "grounding": details, "lexical": lexical,
                           "problems": problems + errs}, ensure_ascii=False)
        return Plan(steps or [], str(parsed.get("reply", "")), raw + "\n" + info, "llm", time.time() - t0,
                    problems + errs, "; ".join(f'{d["target"]} → {d["place"] or "?"}' for d in details), status)

    def _use_cache(self, family: str):
        """プロンプトの系統ごとに KV キャッシュを持ち、共通の前置きを計画をまたいで再利用する。"""
        from transformers import DynamicCache

        fam = self.__dict__.setdefault("_families", {})
        self._cache, self._ids = fam.get(family) or (DynamicCache(), [])

    def _store_cache(self, family: str):
        self._families[family] = (self._cache, self._ids)

    def _greedy(self, text: str, max_tokens: int) -> str:
        """KV キャッシュを使った貪欲生成 (HF の generate と同じ結果)。"""
        stop = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>"),
                self.tok.convert_tokens_to_ids("<|endoftext|>")}
        ids = self.tok(text).input_ids
        logits, _ = self._forward(ids)
        gen = []
        for _ in range(max_tokens):
            t = int(logits[-1].argmax())
            if t in stop:
                break
            gen.append(t)
            logits, _ = self._forward(ids + gen)
        return self.tok.decode(gen)

    def _p_yes(self, scene: str, target: str, obj: str) -> float:
        """「obj は target に当てはまるか」の yes 確率 (yes / no の 2 択で正規化)。"""
        if SPATIAL_WORDS.search(target):
            msgs = [{"role": "system", "content": MATCH_POS_SYSTEM}]
            for sc, desc, qa in MATCH_SHOTS:
                for o, a in qa:
                    msgs += [{"role": "user", "content": f"Objects: {sc}\nDescription: \"{desc}\"\nDoes \"{o}\" match?"},
                             {"role": "assistant", "content": a}]
            msgs.append({"role": "user",
                         "content": f"Objects: {scene}\nDescription: \"{target}\"\nDoes \"{obj}\" match?"})
        else:
            msgs = [{"role": "system", "content": MATCH_SYSTEM}]
            for desc, o, a in NAME_SHOTS:
                msgs += [{"role": "user", "content": _name_question(desc, o)}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": _name_question(target, obj)})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text).input_ids
        logits, _ = self._forward(ids)
        lp = self.torch.log_softmax(logits[-1], -1)
        y, n = float(lp[self._tok_yes]), float(lp[self._tok_no])
        return float(np.exp(y) / (np.exp(y) + np.exp(n)))

    @property
    def _tok_yes(self):
        return self.tok("yes", add_special_tokens=False).input_ids[0]

    @property
    def _tok_no(self):
        return self.tok("no", add_special_tokens=False).input_ids[0]

    def _plan_choice(self, messages, objects, places) -> Plan:
        from transformers import DynamicCache

        t0 = time.time()
        self._cache, self._ids = DynamicCache(), []
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        think = ""
        if self.think:
            text += '{"think": "'
            think = self._greedy_until_quote(text, 80)
            text += think + '", "moves": {'
        else:
            text += '{"moves": {'
        moves, scores = {}, {}
        for i, o in enumerate(objects):
            text += ('' if i == 0 else ', ') + f'"{o}": "'
            stacked_on = set(v for v in moves.values() if v in objects)
            cands = ["stay"] + places + [x for x in objects
                                         if x != o and x not in stacked_on and moves.get(x) != o]
            suffix = '"}' if i == len(objects) - 1 else '", "'
            sc = {c: self._score(text, c, suffix) for c in cands}
            best = max(sc, key=sc.get)
            moves[o], scores[o] = best, {k: round(v, 2) for k, v in sc.items()}
            text += best + '"'
        text += '}, "reply": "'
        reply = self._greedy_until_quote(text, 60)
        steps = _order(moves) or []
        raw = json.dumps({"moves": moves, "reply": reply, "scores": scores}, ensure_ascii=False)
        return Plan(steps, reply, raw, "llm", time.time() - t0, [], think)


# ---------------------------------------------------------------------------- フォールバック
_JA_COLOR = {"赤": "red", "緑": "green", "青": "blue", "黄": "yellow", "オレンジ": "orange", "橙": "orange",
             "紫": "purple", "白": "white", "黒": "black", "ピンク": "pink", "灰": "gray"}
_JA_SHAPE = {"ブロック": "cube", "キューブ": "cube", "積み木": "cube", "ボール": "ball", "玉": "ball", "球": "ball",
             "缶": "cylinder", "円柱": "cylinder"}
_JA_PLACE = {"木箱": "wooden box", "箱": "wooden box", "皿": "white plate"}


def rule_based_plan(instruction: str, objects: list[str], places: list[str]) -> Plan:
    """LLM が使えない/失敗した場合の簡易解析器。「AをBに」形式と「全部」のみ対応。"""
    place = None
    for ja, en in _JA_PLACE.items():
        if ja in instruction and en in places:
            place = en
            break
    colors = [en for ja, en in _JA_COLOR.items() if ja in instruction]
    shapes = [en for ja, en in _JA_SHAPE.items() if ja in instruction]
    cands = [o for o in objects
             if (not colors or any(c in o for c in colors)) and (not shapes or any(s in o for s in shapes))]
    if not ("全部" in instruction or "すべて" in instruction or "全て" in instruction):
        cands = cands[:1]
    place = place or (places[0] if places else None)
    steps = [{"pick": o, "place": place} for o in cands if place]
    return Plan(steps, "（簡易解析で計画しました）", source="rule")
