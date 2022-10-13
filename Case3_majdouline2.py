#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import plotly.express as px
import http.client
import streamlit as st


# # Importeren API dataset van OpenChargeMap

# In[2]:


import http.client
import csv

conn = http.client.HTTPSConnection("api.openchargemap.io")

headers = { 'Content-Type': "application/json" }

conn.request("GET", "/v3/poi/?key=c154fb8a-8bdf-4d6e-9800-3eb95fb347f4&output=csv&countrycode=NL&maxresults=200000&compact=true&verbose=false", headers=headers)
#key: c154fb8a-8bdf-4d6e-9800-3eb95fb347f4
res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))


# In[ ]:





# In[3]:


import io
df_li = pd.read_csv(io.StringIO(data.decode("utf-8")))
df_li.info()


# In[4]:


import http.client

conn = http.client.HTTPSConnection("api.openchargemap.io")

headers = { 'Content-Type': "application/json" }

conn.request("GET", "/v3/poi/?key=c154fb8a-8bdf-4d6e-9800-3eb95fb347f4&output=json&countrycode=NL&maxresults=200000&compact=true&verbose=false", headers=headers)
#key: c154fb8a-8bdf-4d6e-9800-3eb95fb347f4
res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))


# In[5]:


import json

data_json = json.loads(data)

li_adres = []
for item in data_json :
    li_adres.append(item["AddressInfo"])
    

df_li_adres = pd.DataFrame(li_adres)

df_li_adres


# In[6]:


li_connlist = []
for item in data_json :
    li_connlist.append(item["Connections"])
    
li_connectie = []

for m in range(len(li_connlist)):

   # using nested for loop, traversing the inner lists
   for n in range (len(li_connlist[m])):

      # Add each element to the result list
      li_connectie.append(li_connlist[m][n])
        
df_li_connectie = pd.DataFrame(li_connectie)


# In[7]:


df_li_merged = df_li_adres.merge(df_li_connectie, left_on = 'ID', right_on = 'ID')


# We hebben een functie nodig om unieke waarden in een kolom te vinden. Ik wil weten of er meerdere soorten getallen zitten 
# in DistanceUnit. Dat doen we met def unique hieronder.

# In[8]:


def unique(list1):
  
    # een null-lijst/lege lijst initialiseren
    unique_list = []
  
    # door alle elementen heengaan
    for x in list1:
        # controleren of het bestaat in unique_list of niet
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print (x)
        
list1 = df_li_adres["DistanceUnit"]
unique(list1)


# In[9]:


#df_li_adres.columns


# #conclusie: ook dit kolom kan gedropt worden

# In[10]:


# de gegeven lijst bevat duplicaten
mylist =  df_li_adres['ID']

newlist = [] # lege lijst om unieke elementen uit de lijst te bewaren
duplist = [] # lege lijst om de dubbele elementen uit de lijst te bewaren
for i in mylist:
    if i not in newlist:
        newlist.append(i)
    else:
        duplist.append(i) # deze methode vangt de eerste dup's op en voegt ze toe aan de lijst

# het printen van de dups
#print("Lijst duplicates", duplist)
#print("Unieke Item List", newlist) # print uiteindelijke list van unieke waarden


# # CSV files importeren 

# In[11]:


#csv_file= 'laadpaaldata.csv'
#df_1=pd.read_csv(csv_file)
#csv_file2='Elektrische_voertuigen.csv'
#df_2=pd.read_csv(csv_file2)
url = 'https://raw.githubusercontent.com/gomeco/week6/main/laadpaaldata.csv'
df_1 = pd.read_csv(url)


# In[12]:


url1='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen0.csv'
url2='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen1.csv'
url3='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen2.csv'
url4='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen3.csv'
url5='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen4.csv'
url6='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen5.csv'
url7='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen6.csv'
url8='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen7.csv'
url9='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen8.csv'
url10='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen9.csv'
url11='https://raw.githubusercontent.com/gomeco/week6/main/Elektrische_voertuigen10.csv'


file_list=[url1, url2, url3, url4, url5, url6, url7, url8, url9, url10, url11]
  
main_list = []
  
for document in file_list:
    main_list.append(pd.read_csv(document, low_memory=False))
elektrisch = pd.concat(main_list)
df_2 = pd.concat(main_list)


# In[ ]:





# In[13]:


#df_2.info()

# We zien veel kolommen met NA waarden, deze worden hieronder eruit gehaald
# In[14]:


df_2filter=df_2.drop(['Aantal cilinders', 'Laadvermogen', 'Oplegger geremd', 'Aanhangwagen autonoom geremd', 
           'Aanhangwagen middenas geremd', 'Aantal staanplaatsen', 'Afwijkende maximum snelheid',
           'Europese uitvoeringcategorie toevoeging', 'Vervaldatum tachograaf', 'Vervaldatum tachograaf DT',
           'Maximum last onder de vooras(sen) (tezamen)/koppeling', 'Type remsysteem voertuig code',
           'Rupsonderstelconfiguratiecode', 'Wielbasis voertuig minimum', 'Wielbasis voertuig maximum',
           'Lengte voertuig minimum', 'Lengte voertuig maximum', 'Breedte voertuig minimum', 'Breedte voertuig maximum',
           'Hoogte voertuig minimum', 'Hoogte voertuig maximum', 'Massa bedrijfsklaar minimaal', 'Massa bedrijfsklaar maximaal',
           'Technisch toelaatbaar massa koppelpunt', 'Maximum massa technisch maximaal', 'Maximum massa technisch minimaal',
           'Subcategorie Nederland', 'Type gasinstallatie', 'Zuinigheidsclassificatie', 'API Gekentekende_voertuigen_carrosserie',
           'API Gekentekende_voertuigen_carrosserie_specifiek', 'Datum tenaamstelling DT', 'API Gekentekende_voertuigen_voertuigklasse',
                      'API Gekentekende_voertuigen_assen', 'API Gekentekende_voertuigen_brandstof', 'Europese voertuigcategorie toevoeging',
                      'Cilinderinhoud', 'Bruto BPM', 'Eerste kleur', 'Tweede kleur', 'Voertuigsoort'
                     ], axis=1)
st.code(df_2filter)
st.write('De kollommen zonder waardes en betekenis worden gedropt uit de dataFrame')


# In[15]:


#df_2filter.columns


# In[16]:


def unique(list2):
  
    # een null-lijst/lege lijst initialiseren
    unique_list = []
  
    # door alle elementen heengaan
    for x in list2:
        # controleren of het bestaat in unique_list of niet
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print (x)
        
list2 = df_2filter['Tenaamstellen mogelijk']
unique(list2)



# In[17]:


#df_1["ChargeTime"].min()


# In[18]:


#alleen positieve waardem behouden in ChargeTime

L=df_1['ChargeTime']

[x
   for x in L
   if x >= 0
]


# In[19]:


#df_2filter.columns


# In[20]:


#df_2filter['Datum eerste toelating DT']


# In[21]:


df_datum=df_2filter['Datum eerste toelating DT'] = pd.to_datetime(df_2filter['Datum eerste toelating DT'], errors='coerce')
#df_datum


# In[22]:


df_maand= df_datum.dt.month
#df_maand


# In[23]:


#df_jaar omzetten in een dataframe zodat t gemerged kan worden
df_month = pd.DataFrame({'Maand': df_maand})
#df_month


# In[24]:


#extra tabel aanmaken  
df_2filter['Maand'] = df_month

#df_2filter


# # Kaart in Geopandas

# In[25]:


import geopandas as gpd
import folium


# In[26]:


#df_li


# In[27]:


df_li.NumberOfPoints.unique()


# In[28]:


def unique(list3):
  
    # een null-lijst/lege lijst initialiseren
    unique_list = []
  
    # door alle elementen heengaan
    for x in list3:
        # controleren of het bestaat in unique_list of niet
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print (x)
        
list3 = df_li['NumberOfPoints']
#unique(list3)


# In[29]:


df_dropdowncolums=df_li[['StateOrProvince', 'NumberOfPoints']]
#df_dropdowncolums


# In[30]:


def color_producer(type):
    
    if type == 1.0:
        return "Brown"
    if type == 2.0:
        return "green"
    if type == 3.0:
        return "blue"
    if type == 4.0:
        return "aqua"
    if type == 6.0:
        return "red"
    if type == 7.0:
        return "pink"
    if type == 8.0:
        return "orange"
    if type == 9.0:
        return "gray"
    if type == 10.0:
        return "lightskyblue"
    if type == 12.0:
        return "violet"
    if type == 13.0:
        return "olive"
    if type == 14.0:
        return "lightgreen"
    if type == 15.0:
        return "teal"
    if type == 16.0:
        return "gold"
    if type == 18.0:
        return "peru"
    if type == 20.0:
        return "lavender"
    if type == 24.0:
        return "crimson"
    if type == 28.0:
        return "slategray"
    if type == 29.0:
        return "indigo"
    if type == 32.0:
        return "lime"
    if type == 44.0:
        return "tan"
    if type == 72.0:
        return "mediumspringgreen"



# In[31]:


#print(color_producer(df_li.NumberOfPoints[20]))


# In[32]:


def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map


# In[33]:


import streamlit as st
from streamlit_folium import folium_static
import folium


# In[34]:


#Gekleurde punten toevoegen aan map

#"# streamlit-folium"



m = folium.Map(location=[df_li["Latitude"].mean(), df_li["Longitude"].mean()], zoom_start=8, control_scale=True)

for index, locatie_info in df_li.iterrows():
    color= color_producer(df_li.NumberOfPoints.iloc[index])
    folium.CircleMarker([locatie_info['Latitude'], locatie_info['Longitude']], fill=True, 
                        color = color, tooltip='Klik om het aantal laadpunten te zien',radius= 1.75,
                        popup=f"{df_li.NumberOfPoints.iloc[index]}").add_to(m)
#Legenda
add_categorical_legend(m, 'Legenda: aantal laadpunten', 
                       colors=['Brown', 'green', 'blue', 'aqua', 'red','pink', 'orange',
                                'gray', 'lightskyblue', 'violet', 'olive','lightgreen', 'teal',
                                'gold', 'peru', 'lavender', 'crimson','slategray', 'indigo',
                                'lime', 'tan', 'mediumspringgreen'], 
                       labels=[1.0, 2.0, 3.0,4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                              18.0, 20.0, 24.0, 28.0, 29.0, 32.0, 44.0, 72.0])
folium_static(m)


# In[35]:


st.write('In deze kaart zijn de laadpalen weergeven. Met de kleuren word aangegeven hoeveel laadpalen er op 1 plek zijn')


# In[36]:


import plotly.express as px
import plotly.graph_objects as go


# In[37]:


df_dropdowncolums=df_li[['StateOrProvince', 'NumberOfPoints']]

#df_dropdowncolums


# In[38]:


#df_1


# In[39]:


# alles positief maken
df_1["ChargeTime"] = df_1["ChargeTime"].abs()


# In[40]:


# waardes boven de 10 uur verwijderen
df_1= df_1[(df_1['ChargeTime'] <= 10)]


# In[41]:


data=df_1['ChargeTime']

[x
   for x in data
   if x >= 0
]


# In[42]:


import plotly.express as px

fig = px.histogram(x= data, title="Histogram", nbins= 30)
fig.update_xaxes(title_text='Tijd aan de oplader (uur)')
fig.update_yaxes(title_text='Aantal auto')

annotation = {'x': df_1.ChargeTime.mean(), 'y':100, 'showarrow': True, 'arrowhead': 4,
                    'font': {'color': 'black', 'size':10}, 'text': 'Gemiddelde'}
mediaan = {'x': df_1.ChargeTime.median(), 'y':0, 'showarrow': True, 'arrowhead': 4,
                    'font': {'color': 'red', 'size':10}, 'text': 'Mediaan'}

fig.update_layout({'annotations':[annotation, mediaan]})

#fig.show()

st.plotly_chart(fig, use_container_width=True)


# In[43]:


st.write('Dit is een histogram wat laat zien hoeveel tijd de autos aan de laadpaal zitten')


# In[44]:


# mediaan 
df_1['ChargeTime'].median()


# In[45]:


# gemiddelde
df_1['ChargeTime'].mean()


# In[46]:


df_2filter['Datum eerste toelating DT']

df_datum=df_2filter['Datum eerste toelating DT'] = pd.to_datetime(df_2filter['Datum eerste toelating DT'], 
                                                                  errors='coerce')
#df_datum

df_jaar= df_datum.dt.year
#df_jaar

#df_jaar omzetten in een dataframe zodat t gemerged kan worden
df_jaar = pd.DataFrame({'Maand': df_jaar})
#df_jaar

df_2filter['Jaar'] = df_jaar
#df_2filter


# In[47]:


jaar_nieuwe_auto= df_2filter['Jaar'].value_counts()
#jaar_nieuwe_auto.head()


# In[48]:


df= pd.DataFrame(jaar_nieuwe_auto)


# In[49]:


import matplotlib.pyplot as plt


# In[50]:


plt.hist(x= 'Vermogen massarijklaar', data= elektrisch, range= [0.0,0.1])
plt.title("Aantallen vermogen massarijklaar")
plt.xlabel("Vermogen massarijklaar")
plt.ylabel("Aantal")
plt.show()
st.pyplot()


# In[51]:


st.write('Plotje over het vermogen van massarijklaar bij elektrische autos')


# In[52]:


auto=elektrisch.drop(['Aantal cilinders', 'Laadvermogen', 'Oplegger geremd', 'Aanhangwagen autonoom geremd',
           'Aanhangwagen middenas geremd', 'Aantal staanplaatsen', 'Afwijkende maximum snelheid',
           'Europese uitvoeringcategorie toevoeging', 'Vervaldatum tachograaf', 'Vervaldatum tachograaf DT',
           'Maximum last onder de vooras(sen) (tezamen)/koppeling', 'Type remsysteem voertuig code',
           'Rupsonderstelconfiguratiecode', 'Wielbasis voertuig minimum', 'Wielbasis voertuig maximum',
           'Lengte voertuig minimum', 'Lengte voertuig maximum', 'Breedte voertuig minimum', 'Breedte voertuig maximum',
           'Hoogte voertuig minimum', 'Hoogte voertuig maximum', 'Massa bedrijfsklaar minimaal', 'Massa bedrijfsklaar maximaal',
           'Technisch toelaatbaar massa koppelpunt', 'Maximum massa technisch maximaal', 'Maximum massa technisch minimaal',
           'Subcategorie Nederland', 'Type gasinstallatie', 'Zuinigheidsclassificatie', 'API Gekentekende_voertuigen_carrosserie',
           'API Gekentekende_voertuigen_carrosserie_specifiek', 'Datum tenaamstelling DT', 'API Gekentekende_voertuigen_voertuigklasse',
                      'API Gekentekende_voertuigen_assen', 'API Gekentekende_voertuigen_brandstof', 'Europese voertuigcategorie toevoeging',
                      'Cilinderinhoud', 'Bruto BPM'
                     ], axis=1)
#auto.head()


# In[53]:


#auto['Tweede kleur'].unique()


# In[54]:


#auto.info()


# In[55]:


auto['Datum eerste toelating DT']

df_datum=auto['Datum eerste toelating DT'] = pd.to_datetime(auto['Datum eerste toelating DT'], errors='coerce')
#df_datum

df_jaar= df_datum.dt.year
#df_jaar

#df_jaar omzetten in een dataframe zodat t gemerged kan worden
df_jaar = pd.DataFrame({'Maand': df_jaar})
#df_jaar

auto['Jaar'] = df_jaar
#auto


# In[56]:


jaar_nieuwe_auto= auto['Jaar'].value_counts()
#jaar_nieuwe_auto.head()


# In[57]:


df= pd.DataFrame(jaar_nieuwe_auto)
#df


# In[58]:


#df2_new.head()
df2 = df.reset_index(level=0)

df2 = df2.rename(columns={'index': 'jaar', 'Jaar': 'Aantal'})


#df2


# In[59]:


plt.bar(df2['jaar'],df2['Aantal'],align='center', alpha=0.5)
plt.title("Aantallen (voertuigen) per jaar")
plt.xlabel("Jaartal")
plt.ylabel("Aantal")
st.pyplot()


# In[60]:


st.write('In deze plot is te zien dat na 2010 er pas een grote groei was in het bezitten van een elektrische auto')


# In[61]:


from statsmodels.formula.api import ols
l= ols('Aantal~ jaar', data= df2).fit()
#print(l.params)


# In[62]:


# selecteer vanaf 2016 voor een beter model. Dit komt omdat vanaf dat jaar het aantal elektrische auto's toeneemt. 
df2= df2[(df2['jaar'] > 2016)]
#df2


# In[63]:


#pip install seaborn


# In[64]:


import seaborn as sns
# Create a new figure, fig
fig = plt.figure()

sns.regplot(x="jaar",
            y="Aantal",
            data=df2,
            ci=None)


# Show the layered plot
st.pyplot(fig)


# # Voorspellende analyse met Python

# In[65]:


merk_nieuwe_auto= auto['Merk'].value_counts()
#merk_nieuwe_auto.head()


# In[66]:


df_m= pd.DataFrame(merk_nieuwe_auto)
#df_m


# In[67]:


dfm = df_m.reset_index(level=0)

dfm=dfm.rename(columns={'index': "Merk", "Merk": "aantal"})

#dfm


# In[68]:


dfm=pd.DataFrame(dfm)

#dfm.head()


# In[69]:


dfm=dfm[(dfm['aantal']>10000)]
#dfm


# In[70]:


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.barplot(x=dfm["Merk"], y=dfm["aantal"])
plt.setp(ax.get_xticklabels(), rotation=45)
#ax.set_xticklabels(rotation=30)
plt.title("Aantal gekochte elektrische voertuigen")
plt.xlabel("Elektrische voertuigen merknamen")
plt.ylabel("Aantal gekocht")

st.pyplot()


# In[71]:


#from sklearn.linear_model import LogisticRegression

#logmodel=LogisticRegression()

#logmodel.fit(X_train, y_train)

#predicitons =logmodel.predict(X_test)

#from sklearn.metrics import classification_report

#classification_report(y_test, predictions)


# In[72]:


url1='https://raw.githubusercontent.com/gomeco/week6/main/brandstof2021_1.csv'
url2='https://raw.githubusercontent.com/gomeco/week6/main/brandstof2021_2.csv'
url3='https://raw.githubusercontent.com/gomeco/week6/main/brandstof2021_3.csv'
url4='https://raw.githubusercontent.com/gomeco/week6/main/brandstof2021_4.csv'
url5='https://raw.githubusercontent.com/gomeco/week6/main/brandstof2021_5.csv'

file_list=[url1, url2, url3, url4, url5]
  
main_list = []
  
for document in file_list:
    main_list.append(pd.read_csv(document, low_memory=False))
df_brandstof2021 = pd.concat(main_list)


# In[73]:


url1='https://raw.githubusercontent.com/gomeco/week6/main/kenteken2021_1.csv'
url2='https://raw.githubusercontent.com/gomeco/week6/main/kenteken2021_2.csv'
url3='https://raw.githubusercontent.com/gomeco/week6/main/kenteken2021_3.csv'
url4='https://raw.githubusercontent.com/gomeco/week6/main/kenteken2021_4.csv'
url5='https://raw.githubusercontent.com/gomeco/week6/main/kenteken2021_5.csv'

file_list=[url1, url2, url3, url4, url5]
  
main_list = []
  
for document in file_list:
    main_list.append(pd.read_csv(document, low_memory=False))
df_kenteken2021 = pd.concat(main_list)


# In[74]:


df_analyse = df_brandstof2021.merge(df_kenteken2021, left_on='Kenteken', right_on='Kenteken')


# In[75]:


df_analyse.drop('Datum', axis=1)


# In[76]:


df_analyse['Datum'] = pd.to_datetime(df_analyse['Datum eerste toelating DT'])


# In[77]:


df_analyse['Maand'] = df_analyse['Datum'].dt.month


# In[78]:


line_month_merk = df_analyse[['Maand', 'Merk']].value_counts(sort=False).sort_index()
line_brandstof = df_analyse['Brandstof omschrijving'].value_counts(sort=False).sort_index()
line_month_merk_brandstof = df_analyse[['Maand', 'Brandstof omschrijving', 'Merk']].value_counts(sort=False).sort_index()
line_month = df_analyse['Maand'].value_counts(sort=False).sort_index()
line_month_brandstof = df_analyse[['Maand', 'Brandstof omschrijving']].value_counts(sort=False).sort_index()


# In[79]:


import matplotlib.pyplot as plt
import plotly


# In[80]:


fig = px.line(x=line_month.index, y=line_month, title='Aantal auto')
fig.update_xaxes(title_text='Maand')
fig.update_yaxes(title_text='Aantal auto')
st.plotly_chart(fig, use_container_width=True)


# In[81]:


fig = px.bar(x=line_brandstof.index, y=line_brandstof, title='Aantal auto')
fig.update_xaxes(title_text='Brandstof')
fig.update_yaxes(title_text='Aantal auto')
st.plotly_chart(fig, use_container_width=True)

