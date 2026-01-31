# Recommendation System Pipeline

Система рекомендаций на основе двухуровневой архитектуры.

## Структура проекта

```
.
├── src/
│   ├── models/          # Модели первого уровня
│   │   ├── toppop.py
│   │   └── ials.py
│   ├── candidates/      # Генерация кандидатов
│   │   └── generator.py
│   ├── features/        # Построение признаков
│   │   └── builder.py
│   ├── utils/           # Утилиты
│   │   └── metrics.py
│   └── pipeline/        # Пайплайны
│       ├── train.py
│       └── inference.py
├── notebooks/           # Ноутбуки для анализа
└── data/                # Данные

```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

```python
from src.pipeline.train import prepare_data, train_models, generate_candidates, prepare_features

train_stage_1, valid_stage_1, test, items, users = prepare_data()
toppop, ials = train_models(train_stage_1)
cand = generate_candidates(ials, toppop, valid_stage_1['user_id'].unique(), train_stage_1)
X, feature_cols = prepare_features(cand, valid_stage_1, train_stage_1, items, users)
```

