import os
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

    def gmm(self, n_clusters=3): # gaussian mixture model
        numeric_df = self._df.select_dtypes(include=['float64', 'int64'])
        
        # print(numeric_df.head(10))
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(numeric_df)
        labels = gmm.predict(numeric_df)
        self._df['GMM_Cluster'] = labels
        print("LABELS:", labels)
        print(f"GMM clustering completed with {n_clusters} components.")
 
        # plot
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        components = tsne.fit_transform(numeric_df)
        # pca = PCA(n_components=2)
        # components = pca.fit_transform(numeric_df)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=self._df['GMM_Cluster'], palette='Set2')
        plt.title('GMM Clusters (tsne Reduced to 2D)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title="Cluster")
        plt.show()

        columnes_clau = ['job', 'stud_h', 'cesd', 'stai_t', 'qcae_cog', 'amsp', 'health', 'mbi_ex']  # podem escollir qualsevol paràmetre per analitzar
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(columnes_clau, 1):
            plt.subplot(3, 3, i)  
            sns.violinplot(x='GMM_Cluster', y=feature, data=self._df, palette='Set2')  
            plt.title(f'Distribució de {feature} per cluster')

        plt.tight_layout()
        plt.show()

    plt.show()

a = MentalhealthML()
a.load_dataset("dataset.csv")
# print(a.gmm())
a.biaix()