from typing import Optional
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv
from config import Config

load_dotenv()

class ExtractedDrinkInfo(BaseModel):
    drink_type: Optional[str] = Field(
        default=None, description="Loại đồ uống (Ví dụ: Coffee, Trà, Capuchino,...)"
    )
    drink_color: Optional[str] = Field(
        default=None, description="Màu sắc của đồ uống (Ví dụ: nâu, đỏ, xanh,...)"
    )
    container_type: Optional[str] = Field(
        default=None, description="Loại và hình dáng của cốc hoặc ly (Ví dụ: cốc nhựa, cốc thủy tinh,...)"
    )
    ingredients: Optional[str] = Field(
        default=None, description="Thành phần chính của đồ uống (Ví dụ: sữa, đường, trân châu, đá,...)"
    )
    topping: Optional[str] = Field(
        default=None, description="Lớp phủ trên đồ uống, ví dụ: kem, trân châu, thạch, foam sữa,..."
    )
    suitable_for: Optional[str] = Field(
        default=None, description="Đối tượng phù hợp hoặc hoàn cảnh thưởng thức lý tưởng (Ví dụ: cho người yêu thích đồ ngọt, phù hợp ngày lạnh, lý tưởng cho mùa hè,...)"
    )

class LLMExtract:
    @staticmethod
    def llm_extract(*, drink_description: str) -> Optional[ExtractedDrinkInfo]:
        parser = JsonOutputParser(pydantic_object=ExtractedDrinkInfo)

        system_prompt = (
            "Bạn là một trợ lý thông minh, chuyên trích xuất thông tin có cấu trúc từ mô tả đồ uống.\n\n"
            "Nhiệm vụ của bạn là phân tích kỹ nội dung mô tả tôi cung cấp và trích xuất các thông tin sau:\n"
            "1. **drink_type**: Tên loại đồ uống – viết bằng **tiếng Anh và tiếng Việt** \n"
            "2. **drink_color**: Màu sắc của đồ uống\n"
            "3. **container_type**: Hình dáng và kiểu dáng của cốc hoặc ly \n"
            "4. **ingredients**: Thành phần chính \n"
            "5. **topping**: Lớp phủ nếu có \n"
            "6. **suitable_for**: Đối tượng sử dụng hoặc hoàn cảnh thưởng thức lý tưởng\n\n"
            "👉 Yêu cầu:\n"
            "- Tất cả thông tin phải được viết bằng **tiếng Việt**, ngoại trừ `drink_type` là tiếng Anh kèm tiếng Việt.\n"
            "- Nếu không có thông tin, hãy ghi rõ `không có topping`.\n"
            "- Trả lời đúng theo định dạng JSON sau:\n"
            f"{parser.get_format_instructions()}\n"
            "Chỉ trả lời dưới dạng JSON, không kèm giải thích."
        )

        prompt = [
            AIMessage(content=system_prompt),
            HumanMessage(content=f"Mô tả đồ uống:\n{drink_description}")
        ]

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(" GOOGLE_API_KEY không được thiết lập. Kiểm tra file .env.")
            return None

        llm = GoogleGenerativeAI(
            model=Config.llm_model,
            temperature=Config.llm_temperature,
            google_api_key=Config.google_api_key 
        )

        response = llm.invoke(prompt)

        try:
            data = parser.parse(response)
            return ExtractedDrinkInfo(**data)
        except Exception as e:
            print(" Lỗi khi phân tích phản hồi LLM:", e)
            print("Phản hồi thô:\n", response)
            return None