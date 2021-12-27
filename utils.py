import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_common_elements(df, amount=10, ascending=False):

    bar_color = "pink"
    edge_color = "pink"

    figure, axis = plt.subplots(1, 3, figsize=(20, 5))

    for index, attribute in enumerate(
        ["customer_uuid", "product_uuid", "product_category"]
    ):
        xs = df[attribute].value_counts(ascending=ascending)[:amount].index
        ys = df[attribute].value_counts(ascending=ascending)[:amount].values

        axis[index].bar(x=xs, height=ys, color=bar_color, edgecolor=edge_color)
        axis[index].set_title(attribute)
        axis[index].tick_params(labelrotation=75)


def preprocess_and_split(
    data: pd.DataFrame,
    test_size: float,
    minimum_customer_freq=0,
    minimum_product_freq=0,
):

    data["user_id"] = data.customer_uuid.astype("category").cat.codes.values
    data["item_id"] = data.product_uuid.astype("category").cat.codes.values
    data["label"] = 1

    if minimum_customer_freq > 0:
        data = data[
            data.groupby("user_id").user_id.transform(len) > minimum_customer_freq
        ]
    if minimum_product_freq > 0:
        data = data[
            data.groupby("item_id").item_id.transform(len) > minimum_product_freq
        ]

    data = data.sample(frac=1)
    train, test = train_test_split(data, test_size=test_size)

    return train, test


def get_mappings(df):

    user_int_to_uuid = pd.Series(df.customer_uuid.values, index=df.user_id).to_dict()
    user_uuid_to_int = pd.Series(df.user_id.values, index=df.customer_uuid).to_dict()
    product_int_to_uuid = pd.Series(df.product_uuid.values, index=df.item_id).to_dict()
    product_uuid_to_int = pd.Series(df.item_id.values, index=df.product_uuid).to_dict()
    product_uuid_to_category = pd.Series(
        df.product_category.values, index=df.product_uuid
    ).to_dict()

    return (
        user_int_to_uuid,
        user_uuid_to_int,
        product_int_to_uuid,
        product_uuid_to_int,
        product_uuid_to_category,
    )


def mf_model(n_users: int, n_items: int, n_latent_factors: int):

    item_input = keras.layers.Input(shape=[1], name="Item")
    item_embedding = keras.layers.Embedding(
        n_items + 1, n_latent_factors, name="ItemEmbedding"
    )(item_input)
    item_vector = keras.layers.Flatten(name="FlattenItems")(item_embedding)

    user_input = keras.layers.Input(shape=[1], name="User")
    user_embedding = keras.layers.Embedding(
        n_users + 1, n_latent_factors, name="UserEmbedding"
    )(user_input)
    user_vector = keras.layers.Flatten(name="FlattenUsers")(user_embedding)

    prod = keras.layers.dot([item_vector, user_vector], axes=1, name="DotProduct")
    model = keras.Model([user_input, item_input], prod)

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    return model


def train_model(model, train_data: pd.DataFrame, epochs: int):

    history = model.fit(
        [train_data.user_id, train_data.item_id], train_data.label, epochs=epochs
    )

    return history


def recommend(
    customer_id: str,
    item_embeddings,
    user_embeddings,
    number_of_products,
    user_uuid_to_int,
    product_int_to_uuid,
):
    user_id = user_uuid_to_int[customer_id]
    items = user_embeddings[user_id] @ item_embeddings.T
    item_ids = np.argpartition(items, -number_of_products)[-number_of_products:]
    products = [product_int_to_uuid[item_id] for item_id in item_ids]
    return products
