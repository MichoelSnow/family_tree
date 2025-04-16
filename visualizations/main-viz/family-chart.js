import f3 from '../src/index.js'

// Function to save data to JSON file
async function saveData(data) {
    try {
        const response = await fetch('/save-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            throw new Error('Failed to save data');
        }
        const result = await response.json();
        if (result.status === 'success') {
            console.log('Data saved successfully');
        } else {
            console.error('Error saving data:', result.message);
        }
    } catch (error) {
        console.error('Error saving data:', error);
    }
}

function create(data) {
    const f3Chart = f3.createChart('#FamilyChart', data)
        .setTransitionTime(1000)
        .setCardXSpacing(250)
        .setCardYSpacing(150)
        .setOrientationVertical()
        .setSingleParentEmptyCard(true, { label: 'ADD' })

    const f3Card = f3Chart.setCard(f3.CardHtml)
        .setCardDisplay([["first name", "last name"], ["birthday"]])
        .setCardDim({})
        .setMiniTree(true)
        .setStyle('imageRect')
        .setOnHoverPathToMain()

    const f3EditTree = f3Chart.editTree()
        .fixed(true)
        .setFields(["first name", "last name", "birthday", "avatar"])
        .setEditFirst(true)
        .setOnChange(() => {
            // Save data whenever changes occur
            saveData(f3Chart.store.getData());
        })

    f3EditTree.setEdit()

    f3Card.setOnCardClick((e, d) => {
        f3EditTree.open(d)
        if (f3EditTree.isAddingRelative()) return
        f3Card.onCardClickDefault(e, d)
    })

    f3Chart.updateTree({ initial: true })
    f3EditTree.open(f3Chart.getMainDatum())

    f3Chart.updateTree({ initial: true })
}

// Don't need a DOMContentLoaded event listener since modules are automatically deferred
// fetch("./data.json").then(r => r.json()).then(data => create(data));
fetch("/visualizations/main-viz/data.json").then(r => r.json()).then(data => create(data));
