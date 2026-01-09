from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from .runtime import get_catalog, get_choices, get_model


@dataclass
class LaptopInput:
	product_name: str
	company: str
	type_name: str
	inches: float
	cpu_company: str
	cpu_frequency_ghz: float
	ram_gb: float
	memory: str
	weight_kg: float
	opsys: str


FEATURE_COLS = [
	"Company",
	"TypeName",
	"Inches",
	"CPU_Company",
	"CPU_Frequency (GHz)",
	"RAM (GB)",
	"Memory",
	"Weight (kg)",
	"OpSys",
]


def home(request: HttpRequest) -> HttpResponse:
	catalog = get_catalog()
	context = {
		"catalog": catalog,
		"eur_to_idr": settings.EUR_TO_IDR,
	}
	return render(request, "predictor/home.html", context)


def predict(request: HttpRequest) -> HttpResponse:
	choices = get_choices()
	context = {
		"choices": choices,
		"eur_to_idr": settings.EUR_TO_IDR,
	}

	if request.method != "POST":
		return render(request, "predictor/predict.html", context)

	model = get_model()
	if model is None:
		context["error"] = (
			"Model belum tersedia. Jalankan training dulu: "
			"python ml/train.py (di folder laptop_price_project)."
		)
		return render(request, "predictor/predict.html", context)

	try:
		data = LaptopInput(
			product_name=(request.POST.get("product_name") or "").strip(),
			company=(request.POST.get("company") or "").strip(),
			type_name=(request.POST.get("type_name") or "").strip(),
			inches=float(request.POST.get("inches") or 0),
			cpu_company=(request.POST.get("cpu_company") or "").strip(),
			cpu_frequency_ghz=float(request.POST.get("cpu_frequency_ghz") or 0),
			ram_gb=float(request.POST.get("ram_gb") or 0),
			memory=(request.POST.get("memory") or "").strip(),
			weight_kg=float(request.POST.get("weight_kg") or 0),
			opsys=(request.POST.get("opsys") or "").strip(),
		)
	except ValueError:
		context["error"] = "Input tidak valid. Pastikan semua angka diisi dengan benar."
		return render(request, "predictor/predict.html", context)

	X = pd.DataFrame(
		[
			{
				"Company": data.company,
				"TypeName": data.type_name,
				"Inches": data.inches,
				"CPU_Company": data.cpu_company,
				"CPU_Frequency (GHz)": data.cpu_frequency_ghz,
				"RAM (GB)": data.ram_gb,
				"Memory": data.memory,
				"Weight (kg)": data.weight_kg,
				"OpSys": data.opsys,
			}
		],
		columns=FEATURE_COLS,
	)

	pred_eur = float(model.predict(X)[0])
	pred_idr = pred_eur * float(settings.EUR_TO_IDR)

	context.update(
		{
			"submitted": data,
			"pred_eur": pred_eur,
			"pred_idr": pred_idr,
		}
	)
	return render(request, "predictor/predict.html", context)
