from pydantic import BaseModel
class Health(BaseModel):
    ready: bool
    message: str
class ModelInfo(BaseModel):
    default: bool = False
    id: str
    name: str
    available: bool
    reason: str | None = None
    taxonomy: str = 'eurosat10'
    sources: list[str] = []
    categories: list[dict] = []
class Classes(BaseModel):
    classes: list[str]
class Score(BaseModel):
    class_name: str
    similarity: float
    relative_score: float
    description: str = ''
class Prediction(BaseModel):
    model: str
    model_name: str = ''
    predicted_class: str
    top5: list[Score]
    inference_ms: float
class Predictions(BaseModel):
    results: list[Prediction]
    shared_encoding_ms: float
    agreement: bool
class Robustness(BaseModel):
    original_preview: str
    degraded_preview: str
    original: Predictions
    degraded: Predictions
    note: str
class ResearchRow(BaseModel):
    experiment: str
    condition: str
    top1_accuracy: float
    macro_f1: float | None = None
    category: str
    source: str
class Research(BaseModel):
    rows: list[ResearchRow]
    note: str
