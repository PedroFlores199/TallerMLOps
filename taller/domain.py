BRANDS = {
    "Seat": ["Ibiza", "León", "Arona"],
    "Toyota": ["Yaris", "Corolla", "RAV4"],
    "Renault": ["Clio", "Mégane", "Captur"],
    "Volkswagen": ["Polo", "Golf", "Tiguan"],
    "Ford": ["Fiesta", "Focus", "Kuga"],
    "Peugeot": ["208", "308", "3008"],
    "BMW": ["Serie 1", "Serie 3", "X1"],
    "Audi": ["A1", "A3", "Q3"],
}
BRAND_FACTOR = {
    "Seat": 1.0,
    "Toyota": 1.08,
    "Renault": 0.96,
    "Volkswagen": 1.12,
    "Ford": 1.02,
    "Peugeot": 1.0,
    "BMW": 1.5,
    "Audi": 1.42,
}
TASKS = {
    "Pastillas de freno": ("Frenada ruidosa", "Sin DTC", 75.0, 1.2),
    "Sensor ABS": ("Testigo ABS", "C0035", 65.0, 1.1),
    "Termostato": ("Pérdida de refrigerante", "P0217", 95.0, 2.0),
    "Bobina de encendido": ("Ralentí irregular", "P0300", 100.0, 1.0),
    "Turbo": ("Pérdida de potencia", "P0299", 640.0, 5.0),
    "Aire acondicionado": ("Aire sin frío", "P0532", 95.0, 1.6),
    "Embrague": ("Embrague patina", "P0805", 440.0, 6.5),
    "Abolladura": ("Daño de carrocería", "Sin DTC", 65.0, 2.5),
    "Arañazo": ("Daño de pintura", "Sin DTC", 55.0, 1.8),
    "Pieza rota": ("Rotura exterior", "Sin DTC", 180.0, 1.7),
    "Cristal roto": ("Rotura de cristal", "Sin DTC", 260.0, 1.9),
    "Corrosión": ("Corrosión exterior", "Sin DTC", 90.0, 3.0),
    "Pintura desprendida": ("Daño de pintura", "Sin DTC", 70.0, 2.0),
    "Pieza ausente": ("Pieza exterior ausente", "Sin DTC", 220.0, 2.0),
}
PARTS = {
    "Paragolpes": 1.2,
    "Puerta": 1.1,
    "Capó": 1.3,
    "Aleta": 0.85,
    "Faro": 1.4,
    "Cristal": 1.1,
    "Retrovisor": 0.8,
    "Sin determinar": 1.0,
}
REGIONS = {
    "Almería": 46.0,
    "Cádiz": 45.0,
    "Córdoba": 43.0,
    "Granada": 44.0,
    "Huelva": 43.0,
    "Jaén": 42.0,
    "Málaga": 50.0,
    "Sevilla": 49.0,
}
FEATURES = [
    "brand",
    "model",
    "year",
    "kilometers",
    "engine",
    "repair",
    "symptom",
    "dtc",
    "month",
    "severity",
    "part",
    "labor_rate_eur",
    "parts_price_index",
]
CATEGORICAL = ["brand", "model", "engine", "repair", "symptom", "dtc", "part"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
TARGETS = ["cost_eur", "labor_hours"]
