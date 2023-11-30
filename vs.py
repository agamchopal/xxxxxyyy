import spacy
nlp=spacy.load("en_core_web_sm")
import random
from spacy.util import minibatch,compounding
import random
from spacy.util import minibatch,compounding
from pathlib import Path
from pathlib import Path
import streamlit as st
nlp.pipe_names
train=[("Apple is a fruit.", {"entities": [(0, 5, "Fruit")]}),
   ("brinjal", {"entities": [(0, 7, "Vegetable")]}),
       
       ("Vitamin A",{"entities":[(0,9,"Vitamin")]}),
       ("Retinol",{"entities":[(0,7,"Vitamin A")]}),
       ("Retinyl palmitate",{"entities":[(0,17,"Vitamin A")]}),
       ("Beta-Carotene",{"entities":[(0,13,"Pro-VitaminA")]}),
       ("Thiamine chloride hydrochloride",{"entities":[(0,31,"VitaminB1")]}),
       ("Riboflavin",{"entities":[(0,10,"Vitamin b2")]}),
       ("Riboflavin 5’- phosphate sodium",{"entities":[(0,32,"Vitamin b2")]}),
       ("Calcium",{"entities":[(0,7,"Minerals")]}),
       ("Chloride",{"entities":[(0,8,"Minerals")]}),
       ("Copper",{"entities":[(0,6,"Minerals")]}),
       ("Iodine",{"entities":[(0,6,"Minerals")]}),
       ("Iron",{"entities":[(0,4,"Minerals")]}),
       ("Magnesium ",{"entities":[(0,9,"Minerals")]}),
       ("Manganese ",{"entities":[(0,9,"Minerals")]})
,("Molybdenum ",{"entities":[(0,9,"Minerals")]}),
       ("Phosphorous ",{"entities":[(0,11,"Minerals")]}),
       ("Potassium",{"entities":[(0,9,"Minerals")]})
,("Selenium",{"entities":[(0,8,"Minerals")]})
,("Sodium",{"entities":[(0,6,"Minerals")]}),
       ("Boron",{"entities":[(0,5,"Minerals")]}),
       ("L-Histidine",{"entities":[(0,11,"Amino-acid")]}),
       ("L-Histidine hydrochloride",{"entities":[(0,25,"Amino-acid")]}),
       ("L-Isoleucine",{"entities":[(0,12,"Amino-acid")]}),
       ("L-Isoleucine hydrochloride",{"entities":[(0,26,"Amino-acid")]}),
       ("L-Leucine",{"entities":[(0,9,"Amino-acid")]})
,("L-Leucine hydrochloride",{"entities":[(0,23,"Amino-acid")]}),
       ("L-Lysine hydrochloride",{"entities":[(0,22,"Amino-acid")]}),
       ("DL-Methionine",{"entities":[(0,13,"Amino-acid")]}),
       ("L-Cysteine",{"entities":[(0,10,"Amino-acid")]}),
       ("L-Cysteine hydrochloride",{"entities":[(0,24,"Amino-acid")]}),
       ("L-Carnitine",{"entities":[(0,11,"Amino-acid")]}),
       ("L-Carnitine hydrochloride",{"entities":[(0,25,"Amino-acid")]})
,("L-Citruline",{"entities":[(0,11,"Amino-acid")]})
       ,("Adenosine 5-monophosphate ",{"entities":[(0,25,"Nucleotides")]})
       ,("Vitamin A ",{"entities":[(0,9,"Vitamin")]}),
       ("Vitamin B ",{"entities":[(0,9,"Vitamin")]}),
       ("Vitamin C ",{"entities":[(0,9,"Vitamin")]}),
       ("Vitamin B12 ",{"entities":[(0,10,"Vitamin")]}),
       ("Abelmoschus moschatus ",{"entities":[(0,21,"Botanical Ingrdient")]}),
       ("Stem bark ",{"entities":[(0,9,"Botanical Ingrdient")]}),
("Acacia catechu  ",{"entities":[(0,14,"Botanical Ingrdient")]}),
       ("Bacopa monnier  ",{"entities":[(0,14,"Botanical Ingrdient")]}),
       ("Ajuga bracteosa wall  ",{"entities":[(0,20,"Botanical Ingrdient")]}),
       ("Bombax ceiba L  ",{"entities":[(0,14,"Botanical Ingrdient")]}),
("Brassica rapa L  ",{"entities":[(0,15,"Botanical Ingrdient")]}),
       ("Bixa orellana  ",{"entities":[(0,13,"Botanical Ingrdient")]}),
       ("Camellia sinensis ",{"entities":[(0,17,"Botanical Ingrdient")]}),
       ("Carissa carandas L. ",{"entities":[(0,19,"Botanical Ingrdient")]})]
ner=nlp.get_pipe("ner")
for _,annotations in train:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])
disable_pipes=[pipe for pipe in nlp.pipe_names if pipe!=ner]
from spacy.training.example import Example
optimizer = nlp.resume_training()

for iteration in range(100):
    random.shuffle(train)
    losses={}
    batches=minibatch(train,size=compounding(1.0,4.0,1.001))
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)
        
print("Losses",losses)
for text, _ in train:
    doc=nlp(text)
    print("p")
    print("Entities",[(ent.text,ent.label_) for ent in doc.ents])
entity_dict = {}
for text, _ in train:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    for value, label in entities:
        if value.lower() not in entity_dict:
            entity_dict[value.lower()] = [(value, label)]
        else:
            entity_dict[value.lower()].append((value, label))

# Example usage
input_value = "Adenosine 5-monophosphate"
input_value_lower = input_value.lower()

if input_value_lower in entity_dict:
    entities_for_value = entity_dict[input_value_lower]
    print(f"Entities for '{input_value}': {entities_for_value}")
else:
    print(f"No entities found for '{input_value}'.")
st.title("Entity Search")

# User input for value
input_value = st.text_input("Enter a value:")

# Convert input to lowercase for case-insensitive search
input_value_lower = input_value.lower()

# Display entities if input is found in the dictionary
if input_value_lower in entity_dict:
    entities_for_value = entity_dict[input_value_lower]
    st.subheader(f"Entities for '{input_value}':")
    for value, label in entities_for_value:
        st.write(f"- Entity: {value}, Label: {label}")
else:
    st.warning(f"No entities found for '{input_value}'.")



import streamlit as st
LABEL_COLORS = {
    "Fruit": "#FFD700",
    "Vegetable": "#7FFF00",
    "Vitamin": "#00BFFF",
    "Minerals": "#FF6347",
    "Amino-acid": "#D8BFD8",
    "Botanical Ingrdient": "#20B2AA",
    "Nucleotides": "#FFFF00",
    "Pro-VitaminA": "#8A2BE2",
    # Add more labels and colors as needed
}


# Function to highlight entities in the text
def highlight_entities_in_text(text, entities):
    highlighted_text = text
    for entity, label in entities:
        color = LABEL_COLORS.get(label, "#FFFFFF") 
        highlighted_text = highlighted_text.replace(
            entity, f"<span style='background-color: {color}; padding: 2px; border-radius: 4px;'>{entity} ({label})</span>"
        
        ##highlighted_text = highlighted_text.replace(
            #entity, f"<span style='background-color: #FFFF00'>{entity} ({label})</span>"
        )
    return highlighted_text

# Streamlit app
def main():
    st.title("Entity Highlighter App")

    # User input for text (use st.text_area for multiline input)
    input_text = st.text_area("Enter text:")

    if input_text:
        recognized_entities = []
        doc = nlp(input_text)
        recognized_entities.extend((ent.text, ent.label_) for ent in doc.ents)

        # Highlight entities in the input text
        highlighted_output = highlight_entities_in_text(input_text, recognized_entities)

        # Display the highlighted text with recognized entities and labels
        st.markdown(f"**Highlighted Output with Entities and Labels:**")
        st.markdown(f"{highlighted_output}", unsafe_allow_html=True)

        # Your logic for recognizing entities from the input_text (replace with your actual logic)
        #recognized_entities = [("apple", "Fruit"), ("orange", "Fruit")]

        # Highlight entities in the input text
        #highlighted_output = highlight_entities_in_text(input_text, recognized_entities)

        # Display the highlighted text with recognized entities and labels
        #st.markdown(f"**Highlighted Output with Entities and Labels:**")
        #st.markdown(f"{highlighted_output}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
'''def highlight_entities_in_text(text, entities):
    highlighted_text = text
    for entity, label in entities:
        highlighted_text = highlighted_text.replace(
            entity, f"<span style='background-color: #FFFF00'>{entity} ({label})</span>"
        )
    return highlighted_text
def main():
    st.title("Entity Highlighter App")

    # User input for text (use st.text_area for multiline input)
    input_text = st.text_area("Enter text:")

    if st.button("Highlight Entities"):
        # Recognized entities from the trained model
        recognized_entities = []
        doc = nlp(input_text)
        recognized_entities.extend((ent.text, ent.label_) for ent in doc.ents)

        # Highlight entities in the input text
        highlighted_output = highlight_entities_in_text(input_text, recognized_entities)

        # Display the highlighted text with recognized entities and labels
        st.markdown(f"**Highlighted Output with Entities and Labels:**")
        st.markdown(f"{highlighted_output}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()'''


'''from spacy import displacy

# Function to highlight entities in the text
def highlight_entities_in_text1(text, entities):
    highlighted_text = text
    for entity, label in entities:
        highlighted_text = highlighted_text.replace(
            entity, f"<span style='background-color: {LABEL_COLORS[label]}'>{entity} ({label})</span>"
        )
    return highlighted_text

# Load the pre-trained spaCy model and add NER pipe
nlp = spacy.load("en_core_web_sm")

# Define label colors
LABEL_COLORS = {"Fruit": "#FFD700", "Vegetable": "#7FFF00", "Vitamin": "#00BFFF", "Minerals": "#FF6347", "Amino-acid": "#D8BFD8", "Botanical Ingrdient": "#20B2AA", "Nucleotides": "#FFFF00", "Pro-VitaminA": "#8A2BE2"}

# Streamlit app
def main():
    st.title("Entity Highlighter App")

    # User input for text (use st.text_area for multiline input)
    input_text = st.text_area("Enter text:")

    if st.button("Highlightt Entities1"):
        # Recognized entities from the trained model
        recognized_entities = []
        doc = nlp(input_text)
        recognized_entities.extend((ent.text, ent.label_) for ent in doc.ents)

        # Highlight entities in the input text
        highlighted_output = highlight_entities_in_text(input_text, recognized_entities)

        # Display the highlighted text with recognized entities and labels
        st.markdown(f"**Highlighted Output with Entities and Labels:**")
        st.markdown(f"{highlighted_output}", unsafe_allow_html=True)

        # Display the entities using spaCy displacy visualizer
        displacy_rendered = displacy.render(doc, style="ent", options={"colors": LABEL_COLORS})
        st.markdown(displacy_rendered, unsafe_allow_html=True)

if __name__ == "__main__":
    main()'''
