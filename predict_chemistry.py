from collections import Counter

import torch
from PIL import Image
import re
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel
from rdkit import Chem
import json

_BASE_MODEL_PATH = "./models/Qwen3.5-0.8B"
_LORA_MODEL_PATH = "./models/output_chemistry_decimer1/checkpoint-3816"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if _DEVICE == "cuda" else torch.float32

print(f"正在加载化学模型到 {_DEVICE}...")
_PROCESSOR = AutoProcessor.from_pretrained(_BASE_MODEL_PATH, trust_remote_code=True)
_MODEL = Qwen3_5ForConditionalGeneration.from_pretrained(
    _BASE_MODEL_PATH,
    torch_dtype=_DTYPE,
    trust_remote_code=True
)
_MODEL = PeftModel.from_pretrained(_MODEL, _LORA_MODEL_PATH)
_MODEL = _MODEL.to(_DEVICE)
_MODEL.eval()
print("化学模型加载完成！")


def extract_smiles(raw_output: str) -> str:
    smiles = re.sub(r'(<think>.*?</think>)|(.*?assistant)', '', raw_output, flags=re.DOTALL)
    smiles = re.sub(r'^```(?:smiles)?\s*', '', smiles.strip())
    smiles = re.sub(r'\s*```$', '', smiles)
    return smiles.strip()


def postprocess_smiles(smiles: str) -> str:
    """后处理：修复常见语法错误"""
    # 1. 清洗 Markdown
    smiles = re.sub(r'```(?:smiles)?', '', smiles)
    smiles = smiles.replace("$$", "").strip()

    # 2. 修复被拆开的元素（模型可能把 Br 拆成 B r）
    smiles = re.sub(r'B\s+r', 'Br', smiles)
    smiles = re.sub(r'C\s+l', 'Cl', smiles)

    # 3. 修复环闭合（奇数次数字 → 补闭合）
    ring_nums = re.findall(r'\d', smiles)
    count = Counter(ring_nums)
    for num, cnt in count.items():
        if cnt % 2 != 0:
            smiles = smiles + num
            break

    # 4. 修复括号匹配
    open_paren = smiles.count('(')
    close_paren = smiles.count(')')
    if open_paren > close_paren:
        smiles += ')' * (open_paren - close_paren)
    elif close_paren > open_paren:
        # 多余的右括号，删除末尾多余的
        excess = close_paren - open_paren
        for _ in range(excess):
            idx = smiles.rfind(')')
            if idx != -1:
                smiles = smiles[:idx] + smiles[idx + 1:]

    # 5. 修复方括号
    open_bracket = smiles.count('[')
    close_bracket = smiles.count(']')
    if open_bracket > close_bracket:
        smiles += ']' * (open_bracket - close_bracket)

    return smiles.strip()


def compute_confidence(scores, generated_ids):
    confidences = []
    for i, token_id in enumerate(generated_ids[0]):
        token_scores = scores[i][0]
        probs = torch.softmax(token_scores, dim=-1)
        max_prob = probs[token_id.item()].item()
        confidences.append(max_prob)
    if confidences:
        return round(sum(confidences) / len(confidences), 4)
    return 0.0


def generate(image, prompt_text):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    }]
    text = _PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _PROCESSOR(text=[text], images=[image], return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        outputs = _MODEL.generate(
            **inputs,
            max_new_tokens=512,
            output_scores=True,
            return_dict_in_generate=True,
        )
    generated_ids = outputs.sequences
    scores = outputs.scores
    result_text = _PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    result = re.sub(r'(<think>.*?</think>)|(.*?assistant)', '', result_text, flags=re.DOTALL).strip()

    input_len = inputs.input_ids.shape[1]
    new_token_ids = generated_ids[:, input_len:]
    confidence = compute_confidence(scores, new_token_ids)

    return result, confidence


def is_valid_smiles(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def predict(image_path: str) -> dict:
    """第 1 步：图片 → SMILES（提示词和训练一致）"""
    image = Image.open(image_path).convert("RGB")
    prompt = "<image>I have an image of a handwritten chemical structure. Please write out the structure using SMILES format."

    smiles, confidence = generate(image, prompt)
    smiles = extract_smiles(smiles)

    return {"smiles": smiles, "confidence": confidence}


def find_error(image_path: str, wrong_smiles: str) -> dict:
    """第 2 步：找错（提示词和训练一致）"""
    image = Image.open(image_path).convert("RGB")
    prompt = f"<image>I have an image of a handwritten chemical structure and its OCR recognition result. Please help me to detect possible errors in the recognition result and mark the places where errors occur with <error_start> <error_end> and <deleted>.\n\nerroneous formula: {wrong_smiles}\nMarked formula: "

    marked, confidence = generate(image, prompt)

    return {"marked_smiles": marked, "confidence": confidence}


def fix_error(image_path: str, marked_smiles: str) -> dict:
    """第 3 步：改错（提示词和训练一致）"""
    image = Image.open(image_path).convert("RGB")
    prompt = f"<image>I have an image of a handwritten chemical structure and a predicted formula with error tags, correct the formula by modifying the parts marked with <error_start> and <error_end> and inserting content where <deleted> are present. Output the modifications in a single string with the following format: REPLACE:old -> new for errors to be replaced. INSERT:content after position for missing content. DELETE:to_delete for parts to be removed.\n\nMarked formula: {marked_smiles}\nCorrection log: "

    corrected, confidence = generate(image, prompt)

    lines = corrected.strip().split("\n")
    final_smiles = lines[-1].strip() if lines else corrected
    final_smiles = extract_smiles(final_smiles)

    return {"corrected_smiles": final_smiles, "confidence": confidence}


def predict_with_correction(image_path: str) -> dict:
    """完整流程（带调试输出）"""
    # 第 1 步：识别
    result = predict(image_path)
    smiles = result["smiles"]
    print(f"[第1步] 识别SMILES: {smiles}")
    print(f"[第1步] 是否合法: {is_valid_smiles(smiles)}")

    if is_valid_smiles(smiles):
        print("[结果] 第1步直接成功，无需纠错")
        return {
            "smiles": smiles,
            "confidence": result["confidence"],
            "corrected": False
        }

    # 第 2 步：找错
    print(f"[第2步] 开始找错...")
    error_result = find_error(image_path, smiles)
    marked = error_result["marked_smiles"]
    print(f"[第2步] 标记SMILES: {marked}")
    print(f"[第2步] 标记是否和原SMILES相同: {marked == smiles}")

    # 第 3 步：改错
    print(f"[第3步] 开始改错...")
    fix_result = fix_error(image_path, marked)
    corrected = fix_result["corrected_smiles"]
    print(f"[第3步] 修正SMILES: {corrected}")
    print(f"[第3步] 修正后是否合法: {is_valid_smiles(corrected)}")
    print(f"[第3步] 修正是否和原SMILES相同: {corrected == smiles}")

    if is_valid_smiles(corrected):
        print("[结果] 纠错成功，返回修正结果")
        return {
            "smiles": corrected,
            "confidence": fix_result["confidence"],
            "corrected": True
        }
    else:
        print("[结果] 纠错失败，降级返回第1步原始结果")
        smiles = postprocess_smiles(smiles)
        return {
            "smiles": smiles,
            "confidence": result["confidence"],
            "corrected": False
        }

if __name__ == '__main__':
    for i in range(5, 10):
        img_path = f"data/raw/train/DECIMER_HDM_Dataset_Images/CDK_Depict_1_{i}.png"

        print("=" * 50)
        print(f"样本 {i}")

        # 第 1 步：识别
        result = predict(img_path)
        print(f"识别SMILES: {result['smiles']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"是否合法: {is_valid_smiles(result['smiles'])}")

        # 第 2 步：找错
        error_result = find_error(img_path, result['smiles'])
        print(f"标记SMILES: {error_result['marked_smiles']}")
        print(f"置信度: {error_result['confidence']:.2%}")

        # 第 3 步：改错
        fix_result = fix_error(img_path, error_result['marked_smiles'])
        print(f"修正SMILES: {fix_result['corrected_smiles']}")
        print(f"置信度: {fix_result['confidence']:.2%}")
        print(f"修正后是否合法: {is_valid_smiles(fix_result['corrected_smiles'])}")
        print()