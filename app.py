import streamlit as st
from markdownlit import mdlit
import cohere
from cohere.responses.classify import Example
import pandas as pd
from sklearn import model_selection, svm, tree, ensemble, pipeline, preprocessing
import pickle
from PIL import Image

im = Image.open('favicon.ico')
st.set_page_config(layout='wide',page_title='explore co:here',page_icon=im)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def add_data(text,target):
    if text or target is not None:
        return Example(text,target)

def bot(key):
    co = cohere.Client(key)
    with st.form('form'):
        prompt = st.text_input('Enter the promp')
        model = st.selectbox('Choose the model (lite model works faster)',['command-light','command','command-nightly'])
        temperature = st.number_input('Temperature (more higher that value, more creative outcome becomes)',min_value=0.000,max_value=1.000,step=0.20) 
        max_tokens = st.number_input('Length of the output (tokens), Remember more the value of max token you specified, more your api key exhausted quickly',min_value=10)

        
        generate = st.form_submit_button('Generate Summary')
        if generate:
            response = co.generate(prompt=prompt,model=model,temperature=temperature,max_tokens=max_tokens)
            with st.expander('View the output'):
                output = response.generations[0].text
                st.write(output)  

def clasify(key):
    co = cohere.Client(key)
    if 'examples' not in st.session_state:
        st.session_state['examples'] = []
    if 'tests' not in st.session_state:
        st.session_state['tests'] = []
    option = st.radio('You have two options -',['Do you just wanted to try on fewer custom examples','Do you want to train model on a custom data'])
    
    if option == 'Do you just wanted to try on fewer custom examples':
        c1,c2 = st.columns(2)
        with c1:
            text = st.text_area('Add some examples',placeholder='The order came 5 days early')
        with c2:
            target = st.text_input('Add the target like 0/1 or positive/negative')
        add_ex = st.button('Add your data')
        if add_ex:
            st.session_state['examples'].append(add_data(text,target))
            st.toast('Now you can add more')
        with st.expander('View the added data'):
            st.write(st.session_state['examples'])
            
        test_text = st.text_area('Add your data on which you want to test',placeholder='Try to add the text which is context related to your example data')
        add_test = st.button('Add your test data')
        if add_test:
            st.session_state['tests'].append(test_text)
            st.toast('Now you ca add new test data if you want to')
        with st.expander('View your test data'):
            st.write(st.session_state['tests'])
        
        classify_button = st.button('Classify the text')
        if classify_button:    
            classifications = co.classify(model = 'embed-english-v2.0',examples = [Example(i[0],i[1]) for i in st.session_state['examples']],inputs = st.session_state['tests'])
            with st.expander('Results'):
                st.write(classifications.classifications)
                
    if option == 'Do you want to train model on a custom data':
        with st.expander('Please, upload the data having two columns, one is text and another is label. Drop down to see example'):
            st.markdown("""
                        |    | Text                                                                                                                                                                                                                               |   Label |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----:|
|  0 | a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films                                                                                                                         |   1 |
|  1 | apparently reassembled from the cutting room floor of any given daytime soap                                                                                                                                                    |   0 |
|  2 | they presume their audience wo n't sit still for a sociology lesson , however entertainingly presented , so they trot out the conventional science fiction elements of bug eyed monsters and futuristic women in skimpy clothes |   0 |
|  3 | this is a visually stunning rumination on love , memory , history and the war between art and commerce                                                                                                                          |   1 |
|  4 | jonathan parker 's bartleby should have been the be all end all of the modern office anomie films                                                                                                                               |   1 |
|  5 | campanella gets the tone just right funny in the middle of sad in the middle of hopeful                                                                                                                                         |   1 |
                        """)
        
        upload_data = st.file_uploader('Upload CSV file',['csv'])
        if upload_data:
            data = pd.read_csv(upload_data)
            c1,c2 = st.columns(2)
            text_col = None
            with c1:
                text_col = st.selectbox('Specify the text feature',options = list(data.columns))
            with c2:
                remaining = [col for col in data.columns if col not in [text_col]]
                target = st.selectbox('Target Label',remaining)
            mdlit(f'[green]Text Feature[/green] : {text_col} \n [green]Target label[/green] : {target}')
            updated_data = data.dropna(axis=1)
            st.info('Not going to preprocess that much, We are just dropping Nan value consisting rows.')
            models = {'Support Vector Machine':svm.SVC(),'Decision Tree Classifier':tree.DecisionTreeClassifier(),'Random Forest Classifier':ensemble.RandomForestClassifier()}
                
            model = st.selectbox('Select the model',options = list(models.keys()))
            train_button = st.button('Start the training')
            if train_button:
                x_train,x_test,y_train,y_test = model_selection.train_test_split(list(data[text_col]),list(data[target]),test_size=0.25)
                st.toast('Data splitted successfully',icon='🪓🪓')
                
                # Embed the training set
                embeddings_train = co.embed(texts=x_train,model="embed-english-v2.0").embeddings
                # Embed the testing set
                embeddings_test = co.embed(texts=x_test,model="embed-english-v2.0").embeddings
                
                st.toast('Embedding the training and testing done successfully',icon='👍👍')
                
                classifier = pipeline.make_pipeline(preprocessing.StandardScaler(),models[model])
                
                st.toast(f'Pipeline created for {models[model]}',icon='⚗️⚗️')
                
                classifier.fit(embeddings_train,y_train)
                
                st.toast('Model fitted successfully',icon = '🎉🎉')
                
                score = classifier.score(embeddings_test, y_test)
                
                mdlit(f'Score achieved is [blue]{score}[/blue]')
                
                st.write(type(classifier))
                
                st.download_button("Download Model",data=pickle.dumps(classifier),file_name="model.pkl",
)
                
                

def summarize(key):
    co = cohere.Client(key)
    with st.form('form'):
        text = st.text_input('Enter the text you want to summarize')
        model = st.selectbox('Choose the model (lite model works faster)',['command-light','command'])
        temperature = st.number_input('Temperature (more higher that value, more creative outcome becomes)',min_value=1,max_value=5) 
        length = st.selectbox('Select the length of summaries you want',['short','medium','long'])
        format = st.selectbox('In which format you want summary',['paragraph','bullets'])
        extractiveness = st.selectbox('Extractiveness',['low','medium','high'])
        
        generate = st.form_submit_button('Generate Summary')
        if generate:
            response = co.summarize(text=text,model=model,temperature=temperature,length=length,format=format,extractiveness=extractiveness)
            with st.expander('Summary'):
                summary = response.summary
                st.write(summary)   

def main():
    
    mdlit('# Explore [orange]Co:here[/orange]')

    st.sidebar.image('cohere_logo.png')
    
    option = st.sidebar.selectbox('Cohere is availing this tasks',['Summarization','Text Classification','Chatbot'])
    api_key = st.sidebar.text_input('Enter your cohere api key',type='password')
    if api_key:
        if option == 'Summarization':
            summarize(api_key)
        if option == 'Text Classification':
            clasify(api_key)
        if option == 'Chatbot':
            bot(api_key)
    else:
        st.warning('Enter your cohere api key')
        
if __name__ == '__main__':        
    main()
