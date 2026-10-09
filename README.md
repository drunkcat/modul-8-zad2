# Analiza składki ubezpieczeniowej

Prosta aplikacja webowa w Pythonie i Streamlit do szacowania składki ubezpieczeniowej oraz analizy grup klientów. Użytkownik może uzupełnić profil ręcznie albo wpisać jego opis. Po kliknięciu przycisku aplikacja przesyła tekst do OpenAI, aby rozpoznać wartości pól profilu.

## Funkcje

- Szacowanie składki za pomocą wcześniej wytrenowanego modelu regresyjnego.
- Przypisanie użytkownika do grupy podobnych klientów.
- Tabela z charakterystyką grup i informacjami o grupie użytkownika.
- Sugestie zależne od BMI i informacji o paleniu.
- Ręczne uzupełnianie profilu albo rozpoznawanie danych z tekstu.

OpenAI służy wyłącznie do wyodrębniania informacji z opisu. Składkę i grupę obliczają zapisane modele PyCaret. Wartości nieobecne lub niejednoznaczne w tekście pozostają bez zmian.

## Pliki

- app.py – główny plik aplikacji.
- insurance.csv – dane klientów wykorzystywane do charakterystyki grup.
- pipeline_regression.pkl – zapisany model regresyjny.
- pipeline_clustering.pkl – zapisany model klasteryzacji.
- requirements.txt – biblioteki Pythona potrzebne do uruchomienia.

Plik insurance.csv powinien zawierać kolumny: age, sex, bmi, children, smoker, region i charges. Aktualny kod wczytuje CSV z domyślnym separatorem przecinka. Jeśli dane używają średnika, należy zmienić separator w funkcji load_data albo przygotować plik rozdzielany przecinkami.

## Instalacja i uruchomienie lokalne

Wymagane jest środowisko Python 3.11. Utwórz i aktywuj środowisko wirtualne, a następnie zainstaluj zależności:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt

Ustaw klucz API jako zmienną środowiskową:

    $env:OPENAI_API_KEY = "twój-klucz-api"

Uruchom aplikację:

    streamlit run --server.headless=true app.py

Można też przechowywać klucz lokalnie w pliku .streamlit/secrets.toml:

    OPENAI_API_KEY = "twój-klucz-api"

Nie dodawaj klucza ani pliku secrets.toml do repozytorium.

## Wdrożenie w Streamlit Community Cloud

Umieść plik app.py, pliki danych i modeli oraz requirements.txt w repozytorium. Wybierz app.py jako plik wejściowy aplikacji i Python 3.11 jako wersję środowiska. W ustawieniach aplikacji Cloud dodaj sekret:

    OPENAI_API_KEY = "twój-klucz-api"

## Ważne informacje

- Opis tekstowy jest wysyłany do OpenAI dopiero po kliknięciu przycisku rozpoznawania. Nie wpisuj w nim danych identyfikujących. Jeśli wpiszesz wagę i wzrost, aplikacja obliczy BMI na ich podstawie.
- Klucz API przechowuj po stronie serwera jako sekret, a nie w kodzie.
- Wynik predykcji jest demonstracyjnym szacunkiem modelu, nie ofertą ubezpieczeniową ani gwarancją ceny.
- Numery klastrów są etykietami technicznymi; nie oznaczają rankingu klientów.
- Jakość wyników zależy od wcześniej wytrenowanych modeli oraz danych użytych do ich przygotowania (plik insurance.csv).
