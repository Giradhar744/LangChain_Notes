from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model= 'gemini-2.5-flash')

#schema
class Review(BaseModel):
    key_themes: list[str] = Field(description='Write down all the  key themes discussed in the review in a list')
    summary :str = Field( description='A brief summary of the review')
    sentiment: Literal['pos','neg'] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros:Optional[list[str]]= Field(default= None, description="Write down all the pros inside the list")
    cons:Optional[list[str]]= Field(default= None,description = "Write down all the cons inside the list")
    name : Optional[str] = Field(default=None, description='Write the name of the reviewer')

structured_model = model.with_structured_output(Review) #type:ignore

result = structured_model.invoke('''The ASUS Vivobook S14 is a sleek and portable 14-inch laptop designed for students and professionals who want reliable performance in a lightweight form factor. Featuring modern Intel or Snapdragon processors, it delivers smooth day-to-day productivity, fast boot times, and responsive multitasking, while the 16:10 display provides extra vertical space that enhances reading, coding, and browsing. Its 70Wh battery offers impressive backup, making it ideal for long college days or work sessions without constantly plugging in. The laptop also includes useful features such as a Copilot key for quick AI assistance, an IR webcam with a privacy shutter, and a fast NVMe SSD. However, the Vivobook S14 isn’t perfect—its integrated graphics make it unsuitable for gaming or heavy creative workloads, and some users have reported coil whine, heating under load, or display brightness limitations depending on the model. Additionally, RAM is partially soldered in certain variants, affecting long-term upgradability. Despite these drawbacks, the Vivobook S14 remains a strong option for everyday tasks, offering a good balance of performance, battery life, portability, and features. It’s best suited for students, working professionals, and frequent travelers who value efficiency and mobility over high-end graphics performance.
''')

print(result)
# print(result['summary']) #type:ignore
# print(result['sentiment'])#type:ignore

