import os
import pandas as pd

class MentalhealthML():
    def __init__(self, datafile=None):
        self._df = None
        if datafile:
            self.load_dataset(datafile)

    def load_dataset(self, file=None):
        if not os.path.exists(file):
            file = None
            print(f"## El fitxer {file} no existeix.")
        
        if file is None:
            csv_files = []
            for _, _, files in os.walk(os.getcwd()):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(file)
            if csv_files:
                options = "".join([f"{i} : {j}\n" for i, j in enumerate(csv_files)])
                opt = input(f"Quin d'aquest dataset vols fer servir?\n{options} -> ")
                file = csv_files[int(opt)]
        print(f"Carregant {file}...")
        self._df = pd.read_csv(file)
        print(self._df.head(10))

    def cleanse(self):
        ...


# a = MentalhealthML()
# a.load_dataset("datassdfet.csv")