import os
import pickle
import zipfile

import pandas as pd
import requests

from config import DATA_SAVE_PATH, DATASETS, START


class ML1MDataset:
    def __init__(self, dataset_url, dataset_name):
        self.download_url = dataset_url
        self.dataset_name = dataset_name
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.zip_path = os.path.join(self.data_dir, f"{dataset_name}.zip")
        self.dataset_path = os.path.join(self.data_dir, dataset_name)
        self.data_save_path = DATA_SAVE_PATH

    def download_and_extract(self):
        # check if the files are already downloaded
        if os.path.exists(
            os.path.join(self.data_dir, self.dataset_name, "ratings.dat")
        ):
            return
        # downloading the zip file
        response = requests.get(self.download_url)
        with open(self.zip_path, "wb") as f:
            f.write(response.content)
        # extracting the zip file
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)
        # removing the zip file
        os.remove(self.zip_path)

    @staticmethod
    def load_data(path, names):
        # specify the correct encoding and engine to avoid UnicodeDecodeError and ParserWarning for .dat files
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            encoding="latin-1",
            header=None,
            names=names,
        )
        return df

    def preprocess(self):
        for file in os.listdir(self.dataset_path):
            if file == "ratings.dat":
                ratings = self.load_data(
                    os.path.join(self.dataset_path, file),
                    names=["user_id", "movie_id", "rating", "timestamp"],
                )
                ratings.sort_values(by=["timestamp"], inplace=True)
            if file == "movies.dat":
                movies = self.load_data(
                    os.path.join(self.dataset_path, file),
                    names=["movie_id", "title", "genres"],
                )
                # 0 and 1 are reserved for special tokens for Padding and Masking hence we start indexing from 2
                movies_id_mapping = {
                    k: i + START
                    for i, k in enumerate(sorted(list(movies["movie_id"].unique())))
                }
            if file == "users.dat":
                users = self.load_data(
                    os.path.join(self.dataset_path, file),
                    names=["user_id", "gender", "age", "occupation", "zip-code"],
                )

        movies["movie_mapped"] = movies["movie_id"].map(lambda x: movies_id_mapping[x])
        ratings["movie_mapped"] = ratings["movie_id"].map(
            lambda x: movies_id_mapping[x]
        )
        inverse_movies_id_mapping = {v: k for k, v in movies_id_mapping.items()}

        os.makedirs(os.path.join(self.data_save_path, self.dataset_name), exist_ok=True)
        ratings.to_csv(
            os.path.join(self.data_save_path, self.dataset_name, "ratings_mapped.csv"),
            index=False,
        )
        movies.to_csv(
            os.path.join(self.data_save_path, self.dataset_name, "movies_mapped.csv"),
            index=False,
        )
        users.to_csv(
            os.path.join(self.data_save_path, self.dataset_name, "users.csv"),
            index=False,
        )

        with open(
            os.path.join(
                self.data_save_path,
                self.dataset_name,
                "new_to_old_movie_id_mapping.pkl",
            ),
            "wb",
        ) as fp:
            pickle.dump(inverse_movies_id_mapping, fp)


if __name__ == "__main__":
    ml_dataset = "ml-1m"  # look for keys in DATASETS in config.py
    dataset = ML1MDataset(dataset_url=DATASETS[ml_dataset], dataset_name=ml_dataset)
    dataset.download_and_extract()
    dataset.preprocess()
