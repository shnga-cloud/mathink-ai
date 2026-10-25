from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import predict_math
import predict_chemistry
from rdkit import Chem
from rdkit.Chem import Draw


app = FastAPI(title="MathInk API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def render_smiles_to_svg(smiles: str, output_path: str = "chem_output.svg") -> str:
    import re
    svg = ""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            drawer = Draw.MolDraw2DSVG(200, 200)  # 从 500 改成 250
            opts = drawer.drawOptions()
            opts.clearBackground = True
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()

            # 删除背景 rect
            svg = re.sub(r'<rect[^>]*fill:#FFFFFF[^>]*>.*?</rect>', '', svg, flags=re.DOTALL)
            # 黑色 → 白色
            svg = svg.replace("#000000", "#FFFFFF")
            svg = svg.replace("black", "#FFFFFF")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg)
    except Exception as e:
        print(f"渲染失败: {e}")
    return svg

@app.post("/predict")
async def run_predict(
        file: UploadFile = File(...),
        mode: str = Form("math")
):
    print(f"收到请求 - mode: {mode}, 文件名: {file.filename}, 大小: {file.size}")
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        if mode == "chem":
            print("进入化学模式，开始预测...")
            result = predict_chemistry.predict_with_correction(temp_path)
            smiles = result.get("smiles", "")
            confidence = float(result.get("confidence", 0.95))

            svg = render_smiles_to_svg(smiles, "chem_output")
            print(f"SMILES: {smiles}, SVG 长度: {len(svg)}")

            return {
                "code": 200,
                "message": "识别成功",
                "data": {
                    "type": "chem",
                    "smiles": smiles,
                    "svg": svg,
                    "confidence": confidence
                }
            }
        else:
            result = predict_math.predict(temp_path)
            if isinstance(result, dict):
                latex = result.get("latex", "")
                confidence = float(result.get("confidence", 0.95))
            else:
                latex = str(result)
                confidence = 0.95

            return {
                "code": 200,
                "message": "识别成功",
                "data": {
                    "type": "math",
                    "latex": latex,
                    "confidence": confidence
                }
            }
    except Exception as e:
        print(f"错误: {e}")
        return {
            "code": 500,
            "message": str(e),
            "data": None
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)