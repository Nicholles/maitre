#source ./venv/bin/activate  # sh, bash, or zsh
#python
import pandas as pd
data = pd.read_csv("operations.csv",parse_dates=[1],sep= ';',decimal= ',', dayfirst=True)
print(data)

from collections import Counter

def most_common_words(labels):
    words = []
    for lab in labels:
        words += lab.split(" ")
    counter = Counter(words)
    for word in counter.most_common(100):
        print(word)

most_common_words(data['Libelle'].values)

CATEGS = {
	'MEHDI': 'LOYER',
	'ALAFOLIE': 'SOIN',
	'NYA': 'COURSES',
	'ALDI': 'COURSES',
	'EATS': 'RESTAURANT',
	'SNCF': 'TRANSPORT',
	'U': 'COURSES',
	'MERILI': 'PENSION',
	'JEAN': 'PENSION',
	'DGFIP': 'AIDES',
	'CAF': 'AIDES',
    
}
TYPES = {
    'CARTE': 'CARTE',
    'VIREMENT': 'VIREMENT',
    'RETRAIT': 'RETRAIT',
    'PRELEVEMENT': 'PRELEVEMENT',
    'CHEQUE': 'CHEQUE',
}

EXPENSES = [20,200] # Bornes des catégories de dépense : petite, moyenne et grosse
LAST_BALANCE = 108.92 # Solde du compte APRES la dernière opération en date
WEEKEND = ["Saturday","Sunday"] # Jours non travaillés

# Controle des colonnes
#for c in ['date_operation','libelle','debit','credit']:
#   if c not in data.columns:
#        if (c in ['debit','credit'] and 'montant' not in data.columns) or \
#       (c not in ['debit','credit']):
#            msg = "Il vous manque la colonne '{}'. Attention aux majuscules "
#            msg += "et minuscules dans le nom des colonnes!"
#            raise Exception(msg.format(c))

# Suppression des colonnes innutiles
for c in data.columns:
	if c not in ['Date','Libelle','Montant(EUROS)']:
		del data[c]

# Ajout de la colonne 'montant' si besoin
#if 'montant' not in data.columns:
#    data["debit"] = data["debit"].fillna(0)
#    data["credit"] = data["credit"].fillna(0)
#    data["montant"] = data["debit"] + data["credit"]
#    del data["credit"], data["debit"]

# creation de la variable 'solde_avt_ope'
data = data.sort_values("Date")
print(data.dtypes)
amount = data["Montant(EUROS)"].astype(str).astype(float)
balance = amount.cumsum()
print(balance)
balance = list(balance.values)
last_val = balance[-1]
balance = [0] + balance[:-1]
balance = balance - last_val + LAST_BALANCE
data["solde_avt_ope"] = balance

# Assignation des operations a une categorie et a un type
def detect_words(values, dictionary):
    result = []
    for lib in values:
        operation_type = "AUTRE"
        for word, val in dictionary.items():
            if word in lib:
                operation_type = val
        result.append(operation_type)
    return result
data["categ"] = detect_words(data["Libelle"], CATEGS)
data["type"] = detect_words(data["Libelle"], TYPES)

# creation des variables 'tranche_depense' et 'sens'
def expense_slice(value):
    value = -value # Les dépenses sont des nombres négatifs
    if value < 0:
        return "(pas une dépense)"
    elif value < EXPENSES[0]:
        return "petite"
    elif value < EXPENSES[1]:
        return "moyenne"
    else:
        return "grosse"
data["tranche_depense"] = amount.map(expense_slice)
data["sens"] = ["credit" if m > 0 else "debit" for m in amount]

# Creation des autres variables
format_date = pd.to_datetime(data["Date"])
data["annee"] = format_date.map(lambda d: d.year)
data["mois"] = format_date.map(lambda d: d.month)
data["jour"] = format_date.map(lambda d: d.day)
data["jour_sem"] = format_date.map(lambda d: d.day_name)
data["jour_sem_num"] = format_date.map(lambda d: d.weekday()+1)
data["weekend"] = data["jour_sem"].isin(WEEKEND)
data["quart_mois"] = [int((jour-1)*4/31)+1 for jour in data["jour"]]
        
# Enregistrement au format CSV
data.to_csv("operations_enrichies.csv",index=False)
