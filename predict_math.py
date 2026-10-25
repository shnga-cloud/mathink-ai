import torch
from PIL import Image
import re
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel

_BASE_MODEL_PATH = "./models/Qwen3.5-0.8B"
_LORA_MODEL_PATH = "./models/output_tree/checkpoint-6736"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if _DEVICE == "cuda" else torch.float32

print(f"正在加载数学模型到 {_DEVICE}...")
_PROCESSOR = AutoProcessor.from_pretrained(_BASE_MODEL_PATH)
_MODEL = Qwen3_5ForConditionalGeneration.from_pretrained(
    _BASE_MODEL_PATH,
    torch_dtype=_DTYPE,
    trust_remote_code=True
)
_MODEL = PeftModel.from_pretrained(_MODEL, _LORA_MODEL_PATH)
_MODEL = _MODEL.to(_DEVICE)
_MODEL.eval()
print("模型加载完成！")

def extract_latex(raw_output: str) -> str:
    latex = re.sub(r'(<think>.*?</think>)|(.*?assistant)', '', raw_output, flags=re.DOTALL)
    return latex.strip()

def compute_confidence(scores, generated_ids):
    confidences = []
    for i, token_id in enumerate(generated_ids[0]):
        token_scores = scores[i][0]  # (vocab_size,)
        probs = torch.softmax(token_scores, dim=-1)
        max_prob = probs[token_id.item()].item()
        confidences.append(max_prob)
    if confidences:
        return round(sum(confidences) / len(confidences), 4)
    return 0.0

def predict(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",
             "text": "I have an image of a handwritten mathematical expression. "
                     "Please write out the expression of the formula in the image "
                     "using LaTeX format."}
        ]
    }]
    text = _PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _PROCESSOR(text=[text], images=[image], return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        outputs = _MODEL.generate(
            **inputs,
            max_new_tokens=512,
            output_scores=True,
            return_dict_in_generate=True
        )
    generated_ids = outputs.sequences
    scores = outputs.scores
    result_text = _PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    latex = extract_latex(result_text)
    input_len = inputs.input_ids.shape[1]
    new_token_ids = generated_ids[:, input_len:]
    confidence = compute_confidence(scores, new_token_ids)
    return {"latex": latex, "confidence": confidence}

if __name__ == '__main__':
    result = predict("data/raw/test/CROHME/2014/images/18_em_0.jpg")
    print(f"识别结果：{result['latex']}")
    print(f"置信度：{result['confidence']:.2%}")