from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def _create_num_pipeline(num_imputing_strategy: str, model_type: str) -> Pipeline:
    """

    :rtype: object
    """
    if model_type == 'LR':
        numerical_feature_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy=num_imputing_strategy, copy=True, add_indicator=True))
            ]
        )
        return numerical_feature_transformer
    else:
        numerical_feature_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy=num_imputing_strategy, copy=True, add_indicator=True)),
                ('scaler', StandardScaler())
            ]
        )
    return numerical_feature_transformer


def _create_cat_pipeline(cat_imputing_strategy: str) -> Pipeline:
    """

    :param cat_imputing_strategy: 
    :return: 
    """
    categorical_feature_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy=cat_imputing_strategy, fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]
    )
    return categorical_feature_transformer


def create_pipeline(num_imputing_strategy: str, cat_imputing_strategy: str, data: DataFrame,
                    model_type: str = None) -> ColumnTransformer:
    """

    :param model_type: Type of model
    :param num_imputing_strategy: Strategy of imputing numerical values
    :param cat_imputing_strategy: Strategy of imputing categorical values
    :param data: DataFrame with data to process
    :return:
    """
    numerical_feature_transformer = _create_num_pipeline(num_imputing_strategy, model_type)
    categorical_feature_transformer = _create_cat_pipeline(cat_imputing_strategy)
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical_values', numerical_feature_transformer, data.select_dtypes('number').columns),
            ('categorical_values', categorical_feature_transformer, data.select_dtypes('object').columns)
        ]
    )
    return preprocessor
