const API_URL = "http://127.0.0.1:8000/predict";

const DATASET_URL = "../dataset/agaricus-lepiota.data";


const form = document.getElementById("predictionForm");

const predictButton = document.getElementById("predictButton");

const resultBox = document.getElementById("resultBox");

const errorBox = document.getElementById("errorBox");

const exampleStatus = document.getElementById("exampleStatus");

const exampleStatusTitle =
    document.getElementById("exampleStatusTitle");

const exampleStatusText =
    document.getElementById("exampleStatusText");


const fields = [

    "cap_shape",

    "cap_surface",

    "cap_color",

    "bruises",

    "odor",

    "gill_attachment",

    "gill_spacing",

    "gill_size",

    "gill_color",

    "stalk_shape",

    "stalk_root",

    "stalk_surface_above_ring",

    "stalk_surface_below_ring",

    "stalk_color_above_ring",

    "stalk_color_below_ring",

    "veil_type",

    "veil_color",

    "ring_number",

    "ring_type",

    "spore_print_color",

    "population",

    "habitat"

];


let datasetRows = [];

let currentExampleClass = null;


/* =========================================================
   LOAD DATASET
   ========================================================= */

async function loadDataset() {

    if (datasetRows.length > 0) {
        return datasetRows;
    }


    const response = await fetch(DATASET_URL);


    if (!response.ok) {

        throw new Error(
            "Dataset could not be loaded."
        );

    }


    const text = await response.text();


    datasetRows = text

        .trim()

        .split(/\r?\n/)

        .map(row => row.trim())

        .filter(row => row.length > 0)

        .map(row => row.split(","));


    return datasetRows;
}


/* =========================================================
   FILL FORM
   ========================================================= */

function fillForm(row) {

    const classCode = row[0].trim();

    const values = row.slice(1);


    fields.forEach(function(field, index) {

        const element =
            document.getElementById(field);


        if (!element) {
            return;
        }


        let value = values[index];


        if (value === undefined) {
            value = "";
        }


        value = value.trim();


        /*
         * The dataset uses ?
         * for missing stalk-root values.
         */

        if (value === "?") {

            element.value = "?";

        } else {

            element.value = value;

        }

    });


    currentExampleClass = classCode;


    updateExampleStatus(classCode);
}


/* =========================================================
   UPDATE EXAMPLE STATUS
   ========================================================= */

function updateExampleStatus(classCode) {

    exampleStatus.style.display = "block";


    if (classCode === "e") {

        exampleStatus.className =
            "example-status edible-example";


        exampleStatusTitle.textContent =
            "🟢 Edible Dataset Example";


        exampleStatusText.textContent =
            "This real dataset record is labelled Edible.";

    }


    else if (classCode === "p") {

        exampleStatus.className =
            "example-status poisonous-example";


        exampleStatusTitle.textContent =
            "🔴 Poisonous Dataset Example";


        exampleStatusText.textContent =
            "This real dataset record is labelled Poisonous.";

    }

}


/* =========================================================
   RANDOM EXAMPLE
   ========================================================= */

async function loadRandomExample() {

    try {

        const rows = await loadDataset();


        const randomIndex =
            Math.floor(
                Math.random() * rows.length
            );


        const row = rows[randomIndex];


        fillForm(row);


    }

    catch (error) {

        showDatasetError();

    }

}


/* =========================================================
   EDIBLE EXAMPLE
   ========================================================= */

async function loadEdibleExample() {

    try {

        const rows = await loadDataset();


        const edibleRows =
            rows.filter(
                row => row[0].trim() === "e"
            );


        if (edibleRows.length === 0) {

            throw new Error(
                "No edible examples found."
            );

        }


        const randomIndex =
            Math.floor(
                Math.random() * edibleRows.length
            );


        const row =
            edibleRows[randomIndex];


        fillForm(row);


    }

    catch (error) {

        showDatasetError();

    }

}


/* =========================================================
   POISONOUS EXAMPLE
   ========================================================= */

async function loadPoisonousExample() {

    try {

        const rows = await loadDataset();


        const poisonousRows =
            rows.filter(
                row => row[0].trim() === "p"
            );


        if (poisonousRows.length === 0) {

            throw new Error(
                "No poisonous examples found."
            );

        }


        const randomIndex =
            Math.floor(
                Math.random() *
                poisonousRows.length
            );


        const row =
            poisonousRows[randomIndex];


        fillForm(row);


    }

    catch (error) {

        showDatasetError();

    }

}


/* =========================================================
   DATASET ERROR
   ========================================================= */

function showDatasetError() {

    alert(
        "Unable to load the dataset.\n\n" +
        "Please make sure FastAPI is running and " +
        "the dataset file exists."
    );

}


/* =========================================================
   PREDICTION
   ========================================================= */

form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        resultBox.style.display = "none";

        errorBox.style.display = "none";


        predictButton.disabled = true;

        predictButton.textContent =
            "🔄 Predicting...";


        const data = {};


        fields.forEach(function(field) {

            data[field] =
                document.getElementById(field).value;

        });


        try {

            const response =
                await fetch(
                    API_URL,
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(data)

                    }
                );


            const result =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    result.detail ||
                    "Prediction failed"
                );

            }


            /*
             * DISPLAY PREDICTION
             */

            if (result.prediction === "Edible") {

                document.getElementById(
                    "predictionText"
                ).textContent =
                    "🍄 Edible";

                resultBox.className =
                    "result-box edible";

            }

            else {

                document.getElementById(
                    "predictionText"
                ).textContent =
                    "☠️ Poisonous";

                resultBox.className =
                    "result-box poisonous";

            }


            /*
             * CONFIDENCE
             */

            document.getElementById(
                "confidenceText"
            ).textContent =
                "Confidence: " +
                result.confidence +
                "%";


            /*
             * SAFETY MESSAGE
             */

            document.getElementById(
                "messageText"
            ).textContent =
                result.message;


            resultBox.style.display =
                "block";


            /*
             * CHECK AGAINST DATASET CLASS
             */

            if (currentExampleClass !== null) {

                const predictedCode =
                    result.class_code;


                const expectedClass =
                    currentExampleClass;


                if (
                    predictedCode ===
                    expectedClass
                ) {

                    document.getElementById(
                        "messageText"
                    ).textContent +=
                        " Dataset check: prediction matches the original dataset label.";

                }

                else {

                    document.getElementById(
                        "messageText"
                    ).textContent +=
                        " Dataset check: prediction differs from the original dataset label.";

                }

            }

        }

        catch (error) {

            errorBox.textContent =
                "❌ Unable to connect to the prediction API. " +
                "Please make sure FastAPI is running.";

            errorBox.style.display =
                "block";

        }


        predictButton.disabled = false;

        predictButton.textContent =
            "🔍 Predict Mushroom";

    }
);