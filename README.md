# Com enganyar una intel·ligència artificial
Aquest repositori conté el codi font del Treball de Recerca de Roger Masip Montoy, titulat “Com enganyar una intel·ligència artificial”.

- L’objectiu del projecte és analitzar com responen diferents models d’intel·ligència artificial davant preguntes que desafien les seves polítiques.
S’ha desenvolupat un script en Python capaç d’enviar preguntes repetides a un model, classificar les respostes segons una rúbrica (nivells 0–3) i calcular els percentatges de cada tipus de resposta.

- Funcionalitats principals
	•	Enviament automàtic de preguntes al model d’IA.
	•	Classificació de respostes segons nivells de permissivitat o bloqueig.
	•	Aplicació de la tècnica DSN (Don’t Say No).
	•	Exportació dels resultats per a l’anàlisi posterior.
- Llenguatge i llibreries
  Llibreries: requests, json, tqdm, datetime, unicodedata, re, time
  Llenguatge: Python 3
