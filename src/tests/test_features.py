import pandas as pd
from features import preprocess_categoria_produto, KFoldTargetEncoder, ColumnDropper


def test_column_dropper():
    df = pd.DataFrame(
        {
            "data_compra": [1, 2, 3],  # Coluna a ser removida
            "produto": ["A", "B", "C"],  # Coluna a ser removida
            "categoria_produto": ["X", "Y", "Z"],  # Coluna a ser removida
            "score_fraude_modelo": [0.5, 0.7, 0.3],  # Coluna a ser removida
        }
    )

    dropper = ColumnDropper()
    result = dropper.transform(df)

    assert "data_compra" not in result.columns
    assert "produto" not in result.columns
    assert len(result.columns) == 0


def test_kfold_target_encoder():

    data = pd.DataFrame(
        {
            "categoria_produto": ["A", "B", "A", "B", "C"],  # Variável categórica
            "fraude": [0, 1, 0, 1, 0],  # Target
        }
    )

    encoder = KFoldTargetEncoder(colnames="categoria_produto", n_fold=2)

    encoder.fit(data[["categoria_produto"]], data["fraude"])

    result = encoder.transform(data[["categoria_produto"]])

    assert f"categoria_produto_Kfold_Target_Enc" in result.columns
    assert result["categoria_produto_Kfold_Target_Enc"].isnull().sum() == 0

    # Verificando se o valor de encoding está correto
    # (Este valor vai depender da média das categorias no dataset)
    assert (
        result["categoria_produto_Kfold_Target_Enc"].iloc[0]
        == result["categoria_produto_Kfold_Target_Enc"].iloc[1]
    )


def test_preprocess_categoria_produto():

    data = pd.DataFrame(
        {
            "categoria_produto": ["A", "B", "C", "A", "D"],  # Coluna categórica
        }
    )

    result = preprocess_categoria_produto(data, min_threshold=2)

    assert result["categoria_produto"].nunique() == 2
    assert (
        result["categoria_produto"] == ["A", "Outros", "Outros", "A", "Outros"]
    ).all()
