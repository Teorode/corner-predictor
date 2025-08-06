import pandas as pd
import glob

# 1. Znajdź wszystkie pliki CSV w katalogu
csv_files = glob.glob(r"C:\Users\teodor\Desktop\dane_win\szwecja rożne\*.csv")  # zmień ścieżkę, np. "./data/*.csv"

# 2. Wczytaj każdy plik do DataFrame
dfs = [pd.read_csv(f) for f in csv_files]

# 3. Sprawdź spójność kolumn (opcjonalnie)
print("Unikalne zestawy kolumn w plikach:")
for df in dfs:
    print(tuple(df.columns))

# 4. Połącz wszystkie DataFrame w jeden
combined_df = pd.concat(dfs, ignore_index=True)

# 5. Zapisz połączone dane do nowego pliku CSV
combined_df.to_csv(r"C:\Users\teodor\Desktop\dane_win\szwecja rożne\combined_sezony.csv", index=False)

# Podsumowanie
print(f"Połączono {len(dfs)} plików, wynikowy DataFrame ma {combined_df.shape[0]} wierszy i {combined_df.shape[1]} kolumn.")
