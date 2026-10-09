"""
Wersja 04 – analiza opisu tekstowego z użyciem OpenAI.

Wymagane pliki obok app_04.py:
- insurance.csv
- pipeline_regression.pkl
- pipeline_clustering.pkl

Uruchomienie: streamlit run app_04.py

Zainstaluj dodatkowo pakiety openai i pydantic. Klucz API ustaw jako
OPENAI_API_KEY w zmiennej środowiskowej lub w .streamlit/secrets.toml.
"""

from pathlib import Path
import os
from typing import Literal

import pandas as pd
import streamlit as st
from openai import APIConnectionError, OpenAI
from pydantic import BaseModel, Field
from pycaret.regression import load_model, predict_model
from pycaret.clustering import load_model as load_clustering_model



# ŚCIEŻKI I KONFIGURACJA


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "insurance.csv"
REGRESSION_MODEL_PATH = APP_DIR / "pipeline_regression"
CLUSTERING_MODEL_PATH = APP_DIR / "pipeline_clustering"

st.set_page_config(
    page_title="Analiza składki – wersja 04",
    page_icon="📊",
    layout="wide",
)



# FUNKCJE WCZYTUJĄCE DANE I MODELE


class ExtractedProfile(BaseModel):
    """Pola profilu, które LLM może wyodrębnić z opisu."""

    age: int | None = Field(description="Wiek, jeśli został podany.")
    bmi: float | None = Field(description="BMI, jeśli zostało podane.")
    children: int | None = Field(
        description="Liczba dzieci, jeśli została podana."
    )
    sex: Literal["female", "male"] | None = Field(
        description="Płeć tylko wtedy, gdy jest jawnie wskazana."
    )
    smoker: Literal["yes", "no"] | None = Field(
        description="Czy osoba pali: yes lub no, jeśli wynika z opisu."
    )
    region: Literal[
        "southeast", "southwest", "northwest", "northeast"
    ] | None = Field(
        description="Region tylko wtedy, gdy można go jednoznacznie ustalić."
    )


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Wczytuje dane klientów używane do opisu grup."""
    return pd.read_csv(path)


@st.cache_resource
def load_regression_model_cached(path_without_extension: str):
    """Wczytuje model zapisany przez PyCaret."""
    return load_model(path_without_extension)


@st.cache_resource
def load_clustering_model_cached(path: str):
    """Wczytuje model klasteryzacji zapisany przez PyCaret."""
    return load_clustering_model(path)


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    """Tworzy klienta OpenAI po stronie serwera."""
    return OpenAI(api_key=api_key)


def get_api_key() -> str | None:
    """Pobiera klucz z sekretów Streamlit albo zmiennej środowiskowej."""
    key = os.environ.get("OPENAI_API_KEY")
    try:
        key = st.secrets.get("OPENAI_API_KEY") or key
    except Exception:
        pass
    return key


def parse_user_description() -> None:
    """Callback: wyodrębnia pola z tekstu i aktualizuje kontrolki profilu."""
    text = st.session_state.get("user_text", "").strip()
    if not text:
        st.session_state["parse_notice"] = "Wpisz opis przed rozpoznaniem."
        return

    api_key = get_api_key()
    if not api_key:
        st.session_state["parse_notice"] = (
            "Brak klucza API. Ustaw OPENAI_API_KEY w sekretach Streamlit "
            "lub zmiennej środowiskowej serwera."
        )
        return

    try:
        client = get_openai_client(api_key)
        response = client.responses.parse(
            model="gpt-6-luna",
            input=[
                {
                    "role": "system",
                    "content": (
                        "Wyodrębnij z tekstu informacje jawnie podane "
                        "o profilu ubezpieczeniowym. Jeśli nie podano BMI, ale "
                        "podano wagę i wzrost, oblicz BMI na podstawie tych danych."
                        "Nie zgaduj innych brakujących wartości. Wartości niepodane zwróć "
                        "jako null. Tłumacz polskie określenia regionów "
                        "na odpowiedni kod regionu. Płeć, palenie i region "
                        "ustalaj tylko, gdy opis pozwala na jednoznaczne "
                        "rozpoznanie."
                    ),
                },
                {"role": "user", "content": text},
            ],
            text_format=ExtractedProfile,
        )
        extracted = response.output_parsed
        if extracted is None:
            st.session_state["parse_notice"] = (
                "Model nie zwrócił rozpoznanych pól. Sprawdź opis "
                "lub uzupełnij profil ręcznie."
            )
            return

        applied = []
        rejected = []
        numeric_fields = {
            "age": ("profile_age", 18, 100, "Wiek"),
            "bmi": ("profile_bmi", 10.0, 70.0, "BMI"),
            "children": ("profile_children", 0, 10, "Liczba dzieci"),
        }

        for field, (state_key, minimum, maximum, label) in numeric_fields.items():
            value = getattr(extracted, field)
            if value is None:
                continue
            if minimum <= value <= maximum:
                st.session_state[state_key] = value
                applied.append(label)
            else:
                rejected.append(label)

        categorical_fields = {
            "sex": ("profile_sex", "Płeć"),
            "smoker": ("profile_smoker", "Palenie"),
            "region": ("profile_region", "Region"),
        }
        for field, (state_key, label) in categorical_fields.items():
            value = getattr(extracted, field)
            if value is not None:
                st.session_state[state_key] = value
                applied.append(label)

        if applied:
            message = "Rozpoznano i uzupełniono: " + ", ".join(applied) + "."
        else:
            message = "Nie rozpoznano pól profilu; wartości pozostały bez zmian."
        if rejected:
            message += (
                " Poza dozwolonym zakresem (pozostawiono bez zmian): "
                + ", ".join(rejected)
                + "."
            )
        st.session_state["parse_notice"] = message
    except APIConnectionError as error:
        cause = error.__cause__

        print("Przyczyna:", repr(cause))

        if cause is not None:
            detail = f"{type(cause).__name__}: {cause}"
        else:
            detail = str(error)
        st.session_state["parse_notice"] = (
            "Błąd połączenia z OpenAI: " + detail
        )
    except Exception as error:
        st.session_state["parse_notice"] = (
            f"Nie udało się rozpoznać opisu: {error}"
        )


def make_user_profile(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> pd.DataFrame:
    """Buduje pojedynczy rekord w formacie oczekiwanym przez modele."""
    return pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )



# NAGŁÓWEK I KRÓTKI OPIS


st.title("📊 Analiza składki ubezpieczeniowej")
st.write(
    "Wprowadź profil ręcznie lub opisz go tekstem w panelu bocznym, "
    "aby zobaczyć wyniki analizy."
)

with st.expander("O aplikacji"):
    st.write(
        "Aplikacja korzysta z przygotowanego modelu regresji "
        "do oszacowania składki oraz opcjonalnie z modelu "
        "klasteryzacji do wskazania podobnej grupy klientów."
    )
    st.caption(
        "Opis tekstowy jest wysyłany do OpenAI dopiero po kliknięciu "
        "przycisku rozpoznawania. Nie wpisuj w nim danych identyfikujących."
    )



# PANEL BOCZNY – ZMIENNE PROFILU


st.session_state.setdefault("profile_age", 30)
st.session_state.setdefault("profile_sex", "female")
st.session_state.setdefault("profile_bmi", 25.0)
st.session_state.setdefault("profile_children", 0)
st.session_state.setdefault("profile_smoker", "no")
st.session_state.setdefault("profile_region", "southeast")

with st.sidebar:
    st.header("Profil klienta")
    st.write("Zmieniaj wartości i obserwuj aktualizację wyników.")

    age = st.number_input(
        "Wiek",
        min_value=18,
        max_value=100,
        step=1,
        key="profile_age",
    )
    sex = st.selectbox(
        "Płeć",
        ["female", "male"],
        key="profile_sex",
        format_func=lambda value: {
            "female": "Kobieta",
            "male": "Mężczyzna",
        }[value],
    )
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        step=0.1,
        key="profile_bmi",
    )
    children = st.number_input(
        "Liczba dzieci",
        min_value=0,
        max_value=10,
        step=1,
        key="profile_children",
    )
    smoker = st.selectbox(
        "Czy palisz?",
        ["no", "yes"],
        key="profile_smoker",
        format_func=lambda value: {
            "no": "Nie",
            "yes": "Tak",
        }[value],
    )
    region = st.selectbox(
        "Region",
        ["southeast", "southwest", "northwest", "northeast"],
        key="profile_region",
        format_func=lambda value: {
            "southeast": "Południowy Wschód",
            "southwest": "Południowy Zachód",
            "northwest": "Północny Zachód",
            "northeast": "Północny Wschód",
        }[value],
    )

    st.divider()
    user_text = st.text_area(
        "Opis tekstowy (opcjonalnie)",
        placeholder="Np. Jestem mężczyzną, mam 35 lat, 180 cm wzrostu, ważę 75 kg, nie palę i mam dwoje dzieci.",
        max_chars=1000,
        key="user_text",
        help=(
            "Po kliknięciu przycisku opis zostanie wysłany do OpenAI "
            "w celu rozpoznania pól profilu."
        ),
    )
    st.button(
        "Rozpoznaj dane z opisu",
        type="primary",
        width='stretch',
        on_click=parse_user_description,
    )

if st.session_state.get("parse_notice"):
    st.info(st.session_state["parse_notice"])


# Tabela jednego użytkownika jest wejściem dla modeli.
user_profile = make_user_profile(
    age=int(age),
    sex=sex,
    bmi=float(bmi),
    children=int(children),
    smoker=smoker,
    region=region,
)



# WCZYTANIE MODELI

regression_model = None
clustering_model = None
customer_data = None
data_load_error = None
regression_load_error = None
clustering_load_error = None

if DATA_PATH.exists():
    try:
        customer_data = load_data(str(DATA_PATH))
    except Exception as error:
        data_load_error = str(error)
else:
    data_load_error = "Brak pliku insurance.csv."

if REGRESSION_MODEL_PATH.with_suffix(".pkl").exists():
    try:
        regression_model = load_regression_model_cached(
            str(REGRESSION_MODEL_PATH)
        )
    except Exception as error:
        regression_load_error = str(error)
else:
    regression_load_error = "Brak pliku pipeline_regression.pkl."

if CLUSTERING_MODEL_PATH.with_suffix(".pkl").exists():
    try:
        clustering_model = load_clustering_model_cached(
            str(CLUSTERING_MODEL_PATH)
        )
    except Exception as error:
        clustering_load_error = str(error)
else:
    clustering_load_error = "Brak pliku pipeline_clustering.pkl."



# GŁÓWNA CZĘŚĆ – WYNIKI REAGUJĄCE NA PANEL BOCZNY


predicted_charge = None
prediction_error = None
if regression_model is not None:
    try:
        prediction = predict_model(
            regression_model,
            data=user_profile,
        )
        predicted_charge = float(
            prediction["prediction_label"].iloc[0]
        )
    except Exception as error:
        prediction_error = str(error)

with st.container(border=True):
    st.markdown("# 💰 Przewidywana składka")
    if predicted_charge is not None:
        st.metric("Szacunek modelu", f"${predicted_charge:,.2f}")
    elif prediction_error:
        st.error(f"Nie udało się obliczyć predykcji: {prediction_error}")
    else:
        st.info(regression_load_error)

st.header("💡 Sugestie")
suggestion_col1, suggestion_col2 = st.columns(2)

with suggestion_col1:
    st.subheader("🚭 Palenie")
    if smoker == "yes":
        st.write(
            "W profilu zaznaczono palenie. Rozważ rzucenie palenia "
            "i skonsultuj dostępne wsparcie ze specjalistą."
        )
    else:
        st.write(
            "W profilu nie zaznaczono palenia. Utrzymuj ten korzystny "
            "nawyk i unikaj rozpoczynania palenia."
        )

with suggestion_col2:
    st.subheader("⚖️ BMI")
    if bmi >= 30:
        st.write(
            f"Twoje BMI wynosi {bmi:.1f}. Zadbaj o zdrową masę ciała "
            "poprzez stopniowe zmiany nawyków; w razie potrzeby "
            "skonsultuj je ze specjalistą."
        )
    elif bmi >= 25:
        st.write(
            f"Twoje BMI wynosi {bmi:.1f}. Zwróć uwagę na zdrowe "
            "odżywianie i regularną aktywność, aby wspierać prawidłową "
            "masę ciała."
        )
    else:
        st.write(
            f"Twoje BMI wynosi {bmi:.1f}. Utrzymuj zdrowe nawyki, "
            "zbilansowane odżywianie i regularny ruch."
        )

user_cluster_id = None
if clustering_model is not None:
    try:
        user_cluster_id = int(clustering_model.predict(user_profile)[0])
    except Exception as error:
        st.error(f"Nie udało się przypisać grupy: {error}")
else:
    st.info(clustering_load_error)

st.header("Charakterystyka grup klientów")
if customer_data is None:
    st.info(f"Nie można przygotować charakterystyki: {data_load_error}")
elif clustering_model is None:
    st.info("Nie można przygotować charakterystyki bez modelu klasteryzacji.")
else:
    try:
        cluster_features = [
            "age", "sex", "bmi", "children", "smoker", "region",
        ]
        cluster_data = customer_data.copy()
        cluster_data["Cluster"] = clustering_model.predict(
            cluster_data[cluster_features]
        )
        cluster_summary = (
            cluster_data.groupby("Cluster")
            .agg(
                Liczba_osób=("age", "count"),
                Średni_wiek=("age", "mean"),
                Średnie_BMI=("bmi", "mean"),
                Średnia_składka=("charges", "mean"),
                Średnia_liczba_dzieci=("children", "mean"),
            )
            .round(2)
            .rename(
                columns={
                    "Liczba_osób": "Liczba osób",
                    "Średni_wiek": "Średni wiek",
                    "Średnie_BMI": "Średnie BMI",
                    "Średnia_składka": "Średnia składka",
                    "Średnia_liczba_dzieci": "Średnia liczba dzieci",
                }
            )
        )
        cluster_summary.index.name = "Grupa"
        st.dataframe(
            cluster_summary,
            width='stretch',
        )

        if user_cluster_id is not None:
            with st.container(border=True):
                st.markdown(
                    f"### Zostałeś przypisany do grupy: {user_cluster_id}"
                )

            user_group_data = cluster_data[
                cluster_data["Cluster"] == user_cluster_id
            ]
            if not user_group_data.empty:
                st.subheader(f"Informacje o grupie {user_cluster_id}")
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Liczba osób", len(user_group_data))
                with info_col2:
                    st.metric(
                        "Średni wiek",
                        f"{user_group_data['age'].mean():.1f}",
                    )
                with info_col3:
                    st.metric(
                        "Średnia składka",
                        f"${user_group_data['charges'].mean():,.2f}",
                    )
            else:
                st.info("W danych nie znaleziono osób z tej grupy.")
    except Exception as error:
        st.error(f"Nie udało się przygotować charakterystyki grup: {error}")

st.header("Opis tekstowy")
if user_text.strip():
    st.write(user_text)
else:
    st.caption("Nie wpisano opisu tekstowego.")

st.caption(
    "To wersja robocza. Wynik modelu jest szacunkiem i nie stanowi oferty "
    "ubezpieczeniowej. Po kliknięciu przycisku opis tekstowy jest wysyłany "
    "do OpenAI w celu wyodrębnienia pól profilu."
)
