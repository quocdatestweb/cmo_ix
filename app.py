from flask import Flask, request, jsonify,render_template, json,session
from flask import Markup
from flask_cors import CORS, cross_origin
from keras import backend as K
from joblib import Parallel, delayed
import joblib
import os
import json
import pickle
import numpy as np
from scipy import stats
import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import string
import pandas as pd
import glob
import scipy.sparse as sparse
import keras
import cv2
from pymongo import MongoClient
from pprint import pprint
# from keras_efficientnets import custom_object
from keras.utils.generic_utils import get_custom_objects
from keras_efficientnets import EfficientNetB2
# import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding='utf-8')
import werkzeug
import random
from fertilizer import fertilizer_dic
import bcrypt

app = Flask(__name__)
app.secret_key = "testappplant"

cors = CORS(app)
client = MongoClient('mongodb+srv://admin:admin@cluster0.iey5z.mongodb.net/test')
db = client['plant_disease']


@app.route("/")
def home_view():
        return render_template('index.html')

@app.route("/plant_predict")
def plant_predict():
     # Retrieve session data
    username = session.get("username")
    imgsrc = session.get("imgsrc")
    
        # Display session data
    if username:
        return render_template('plant_predict.html', username = username, imgsrc = imgsrc)
    else:
        return render_template('index.html')

@app.route("/plant_recom")
def plant_recom():
       # Retrieve session data
    username = session.get("username")
    imgsrc = session.get("imgsrc")
        # Display session data
    if username:
        return render_template('plant_recom.html', username =username, imgsrc = imgsrc)
    else:
        return render_template('index.html')
    

@app.route("/forum")
def forum():
       # Retrieve session data
    username = session.get("username")
    imgsrc = session.get("imgsrc")
        # Display session data
    if username:
        return render_template('forum.html', username =username, imgsrc = imgsrc)
    else:
        return render_template('index.html')
    
@app.route("/predicts")
def predicts():
       # Retrieve session data
    username = session.get("username")
    
    imgsrc = session.get("imgsrc")
        # Display session data
    if username:
        return render_template('predicts.html', username = username, imgsrc = imgsrc)
    else:
        return render_template('index.html')
    
@app.route("/fertilizer")
def fertilizer():
        # Retrieve session data
    username = session.get("username")
    imgsrc = session.get("imgsrc")
        # Display session data
    if username:
        return render_template('fertilizer.html', username = username, imgsrc = imgsrc)
    else:
        return render_template('index.html')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add("Access-Control-Allow-Headers",
                         "Origin, X-Requested-With, Content-Type, Accept")
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# Loading all Crop Recommendation Models
crop_xgb_pipeline = pickle.load(
    open("./models/crop_recommendation/xgb_pipeline.pkl", "rb")
)
crop_rf_pipeline = pickle.load(
    open("./models/crop_recommendation/rf_pipeline.pkl", "rb")
)
# crop_knn_pipeline = pickle.load(
#     open("./models/crop_recommendation/knn_pipeline.pkl", "rb")
# )
crop_label_dict = pickle.load(
    open("./models/crop_recommendation/label_dictionary.pkl", "rb")
)


# Loading all Fertilizer Recommendation Models
fertilizer_xgb_pipeline = pickle.load(
    open("./models/fertilizer_recommendation/xgb_pipeline.pkl", "rb")
)
fertilizer_rf_pipeline = pickle.load(
    open("./models/fertilizer_recommendation/rf_pipeline.pkl", "rb"))
fertilizer_svm_pipeline = pickle.load(
    open("./models/fertilizer_recommendation/svm_pipeline.pkl", "rb"))
fertilizer_label_dict = pickle.load(
    open("./models/fertilizer_recommendation/fertname_dict.pkl", "rb"))
soiltype_label_dict = pickle.load(
    open("./models/fertilizer_recommendation/soiltype_dict.pkl", "rb"))
croptype_label_dict = pickle.load(
    open("./models/fertilizer_recommendation/croptype_dict.pkl", "rb"))
crop_label_name_dict = {}
for crop_value in croptype_label_dict:

    crop_label_name_dict[croptype_label_dict[crop_value]] = crop_value
    
    soil_label_dict = {}
    
    for soil_value in soiltype_label_dict:
        soil_label_dict[soiltype_label_dict[soil_value]] = soil_value

    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError

    def crop_prediction(input_data):
        prediction_data = {"xgb_model_prediction": crop_label_dict[crop_xgb_pipeline.predict(input_data)[0]],
                           "xgb_model_probability": max(crop_xgb_pipeline.predict_proba(input_data)[0]) * 100,
                           "rf_model_prediction": crop_label_dict[crop_rf_pipeline.predict(input_data)[0]],
                           "rf_model_probability": max(crop_rf_pipeline.predict_proba(input_data)[0]) * 100,}
                        #    "knn_model_prediction": crop_label_dict[crop_knn_pipeline.predict(input_data)[0]],
                        #    "knn_model_probability": max(crop_knn_pipeline.predict_proba(input_data)[0]) * 100, }
        all_predictions = [
            prediction_data["xgb_model_prediction"],
            prediction_data["rf_model_prediction"],
            # prediction_data["knn_model_prediction"],
        ]

        all_probs = [
            prediction_data["xgb_model_probability"],
            prediction_data["rf_model_probability"],
            # prediction_data["knn_model_probability"],
        ]

        if len(set(all_predictions)) == len(all_predictions):
            prediction_data["final_prediction"] = all_predictions[all_probs.index(
                max(all_probs))]
        else:
            prediction_data["final_prediction"] = stats.mode(all_predictions)[0][0]

        return prediction_data


def fertilizer_prediction(input_data):
    prediction_data = {
        "xgb_model_prediction": fertilizer_label_dict[
            fertilizer_xgb_pipeline.predict(input_data)[0]
        ],
        "xgb_model_probability": max(
            fertilizer_xgb_pipeline.predict_proba(input_data)[0]
        )
        * 100,
        "rf_model_prediction": fertilizer_label_dict[
            fertilizer_rf_pipeline.predict(input_data)[0]
        ],
        "rf_model_probability": max(fertilizer_rf_pipeline.predict_proba(input_data)[0])
        * 100,
        "svm_model_prediction": fertilizer_label_dict[
            fertilizer_svm_pipeline.predict(input_data)[0]
        ],
        "svm_model_probability": max(
            fertilizer_svm_pipeline.predict_proba(input_data)[0]
        )
        * 100,
    }

    all_predictions = [
        prediction_data["xgb_model_prediction"],
        prediction_data["rf_model_prediction"],
        prediction_data["svm_model_prediction"],
    ]

    all_probs = [
        prediction_data["xgb_model_probability"],
        prediction_data["rf_model_probability"],
        prediction_data["svm_model_probability"],
    ]

    if len(set(all_predictions)) == len(all_predictions):
        prediction_data["final_prediction"] = all_predictions[all_probs.index(
            max(all_probs))]
    else:
        prediction_data["final_prediction"] = stats.mode(all_predictions)[0][0]

    return prediction_data


# @app.route("/predict_crop", methods=["GET", "POST"])
# def predictcrop():
#     try:
#         if request.method == "POST":
#             form_values = request.form.to_dict()
#             column_names = ["N", "P", "K", "temperature",
#                             "humidity", "ph", "rainfall"]
#             print(form_values["N"])
#             print(form_values["P"])
#             input_data = np.asarray([float(form_values[i].strip()) for i in column_names]).reshape(
#                 1, -1
#             )
#             prediction_data = crop_prediction(input_data)
#             json_obj = json.dumps(prediction_data, default=convert)
#             return json_obj
#     except:
#         return json.dumps({"error": "Hãy kiểm tra lại dữ liệu bạn nhập"}, default=convert)
    
# @app.route("/predict_crop",  methods=["GET", "POST"])
# def predictcrop():
#     if request.method == 'POST':
#         try:
#             N = int(request.form['N'])
#             P = int(request.form['P'])
#             K = int(request.form['K'])
#             ph = float(request.form['ph'])
#             humidity = int(request.form['humidity'])
#             temperature = int(request.form['temperature'])
#             rainfall = float(request.form['rainfall'])
#             # print(N,P,K,ph,humidity,temperature,rainfall)

        
#             # form_values = request.form.to_dict()
#             # column_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
#             input_data = np.array([N,P,K,temperature,humidity,ph,rainfall])
#             print(input_data)
#             prediction_data = crop_prediction(input_data)
#             json_obj = json.dumps(prediction_data, default=convert)
#             print(prediction_data)
#             return json_obj
#         except:
#             return json.dumps({"error": "Hãy kiểm tra lại dữ liệu bạn nhập"}, default=convert)


@app.route("/predict_fertilizer", methods=["GET", "POST"])
def predictfertilizer():
    try:
        if request.method == "POST":
            form_values = request.form.to_dict()
            column_names = [
                "nitrogen",
                "potassium",
                "phosphorous",
                "soil_type",
                "crop_type",
                "temperature",
                "humidity",
                "moisture",
            ]
            crop_dict = {"ngô": "Maize",
                         "mía": "Sugarcane",
                         "bông": "Cotton",
                         "thuốc lá": "Tobacco",
                         "lúa": "Paddy",
                         "lúa mạch": "Barley",
                         "lúa mì": "Wheat",
                         "kê": "Millets",
                         "hạt dầu": "Oid seeds",
                         "bột giấy": "Pulses",
                         "hạt xay": "Ground nuts"}

            soil_dict = {"đất cát pha": "Sandy",
                         "đất pha sét": "Loamy",
                         "đất đen": "Black",
                         "đất đỏ": "Red",
                         "đất sét": "Clayey"}

            form_values["crop_type"] = crop_dict[form_values["crop_type"]]
            form_values["soil_type"] = soil_dict[form_values["soil_type"]]

            for key in form_values:
                form_values[key] = form_values[key].strip()

            form_values["crop_type"] = crop_label_name_dict[form_values["crop_type"]]
            form_values["soil_type"] = soil_label_dict[form_values["soil_type"]]
            input_data = np.asarray([float(form_values[i]) for i in column_names]).reshape(
                1, -1
            )
            prediction_data = fertilizer_prediction(input_data)
            json_obj = json.dumps(prediction_data, default=convert)
            return json_obj
    except:
        return json.dumps({"error": "Please Enter Valid Data"}, default=convert)


data = ['Bệnh vảy Táo',
        'Bệnh thối đen trên Táo',
        'Bệnh gỉ sắt trên Táo',
        'Táo khỏe mạnh',
        'Bệnh héo xanh do vi khuẩn trên Chuối',
        'Bệnh vệt lá đen trên Chuối',
        'Chuối khỏe mạnh',
        'Việt quất khỏe mạnh',
        'Cherry khỏe mạnh',
        'Bệnh phấn trắng trên Cherry',
        'Bệnh đốm đen trên Cam Quýt',
        'Bệnh thối nhũng trên Cam Quýt',
        'Cam Quýt khỏe mạnh',
        'Bệnh đốm lá xám trên Ngô',
        'Bệnh gỉ sắt trên Ngô',
        'Ngô khỏe mạnh',
        'Bệnh cháy lá Ngô Bắc',
        'Dưa Leo khỏe mạnh',
        'Bệnh sương mai trên Dưa Leo',
        'Bệnh đốm nâu cành trên Thanh Long',
        'Bệnh thối đen trên Nho',
        'Bệnh sởi đen trên Nho',
        'Nho khỏe mạnh',
        'Bệnh cháy lá trên Nho',
        'Bệnh vàng lá gân xanh trên Cam',
        'Bệnh đốm do vi khuẩn trên Đào',
        'Đào khỏe mạnh',
        'Bệnh đốm do vi khuẩn trên Tiêu',
        'Tiêu khỏe mạnh',
        'Bệnh héo sớm trên Khoai Tây',
        'Khoai Tây khỏe mạnh',
        'Bệnh héo muộn trên Khoai Tây',
        'Dâu Tằm khỏe mạnh',
        'Bệnh đốm vằn trên Lúa',
        'Đậu Nành khỏe mạnh',
        'Bệnh phấn trắng trên Bí',
        'Dâu khỏe mạnh',
        'Bệnh lá cháy xém trên Dâu',
        'Bệnh đốm do vi khuẩn trên Cà Chua',
        'Bệnh héo sớm trên Cà Chua',
        'Cà Chua khỏe mạnh',
        'Bệnh héo muộn trên Cà Chua',
        'Bệnh mốc lá trên Cà Chua',
        'Bệnh đốm Septoria trên Cà Chua',
        'Bệnh đốm nhện trên Cà Chua',
        'Bệnh đốm đen trên Cà Chua',
        'Bệnh khảm trên Cà Chua',
        'Bệnh vàng lá xoăn trên Cà Chua']


@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    # try:
    if request.method == 'POST':
        img_path = ""
        # for web
        form_values = request.form.to_dict()
        for i, j in form_values.items():
            img_path = img_path + j

        # for mobile
        if (img_path == ""):
            imagefile = request.files["image"]
            # Getting file name of the image using werkzeug library
            filename = werkzeug.utils.secure_filename(imagefile.filename)
            # Saving the image in images Directory
            imagefile.save("static/image/" + filename)
            img_path = filename

        if (img_path != ""):
            print('Image received', img_path)
            img = cv2.imread('static/image/' + img_path)
            img = cv2.resize(img/255, (224, 224))
            img = np.reshape(img, [1, 224, 224, 3])
            K.clear_session()
            model = keras.models.load_model('efficientnetb2.h5')

            def get_predict(image):
                img = cv2.imread(image)
                img = cv2.resize(img/255, (224, 224))
                img = np.reshape(img, [1, 224, 224, 3])
                return model.predict(img)

            def get_feature_img(image):
                img = cv2.imread(image)
                img = cv2.resize(img/255, (224, 224))
                img = np.reshape(img, [1, 224, 224, 3])
                inter_model = keras.Model(
                    model.input, model.get_layer(index=2).output)
                return inter_model.predict(img)[0]

            def rc_disease_similarity(image, model):
                feature_image_ = get_feature_img(image)
                doc = name_all_clean[np.argmax(get_predict(image))]
                if (doc not in name_healthy_clean):
                    query_vector1 = tfidf_vectorizer.transform([doc])
                    query_vector = sparse.hstack(
                        (query_vector1, feature_image_))
                    arr = max(cosine_similarity(query_vector, df_new))

                    arr_sort = arr.copy()
                    sort_indices = np.argsort(arr_sort)[::-1]
                #     arr_sort[:] = arr_sort[sort_indices]
                    out = [doc]
                    treat = []
                    i = 0
                    while len(out) < 10:
                        if name_all_clean[sort_indices[i]] not in out and name_all_clean[sort_indices[i]] not in name_healthy_clean:
                            out.append(name_all_clean[sort_indices[i]])
                            # print(treatment[i])
                            treat.append(treatment[i])
                        i += 1
                    return [out[1:], treat]
                else:
                    return None

            def clean_document(doc):
                text_clean = "".join(
                    [i.lower() for i in doc if i not in string.punctuation])
                return text_clean
            classes = model.predict(img)
            a = np.argmax(classes)
            # connect database
           
            collection_all = db['name_plant']
            collection_healthy = db['name_plant_healthy']
            collection_treatment = db['plant_dis']
            name_all = []
            name_healthy = []
            treatment = []
            symptom = []
            for i in collection_all.find():
                k = 0
                for j in collection_treatment.find():
                    if i['name'] == j['label_vi']:
                        k = 1
                        symptom.append(j['symptom'])
                        treatment.append(j['treatment'])
                        break
                if k == 0:
                    symptom.append('')
                    treatment.append('')
                name_all.append(i['name'])
            for i in collection_healthy.find():
                name_healthy.append(i['name'])
            # tfdif
            name_all_clean = [clean_document(i) for i in name_all]
            name_healthy_clean = [clean_document(i) for i in name_healthy]

            stopwords_list = ['bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ',
                              'chưa', 'có', 'có_thể', 'cứ', 'cùng', 'cũng', 'đã', 'đang', 'để', 'do', 'đó',
                              'được', 'gì', 'khi', 'không', 'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'này', 'nên',
                              'nếu', 'ngay', 'nhiều', 'như', 'nhưng', 'những', 'nơi', 'nữa', 'phải', 'qua', 'ra',
                              'rằng', 'rất', 'rồi', 'sau', 'sẽ', 'theo', 'thì', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy', 'vì', 'việc', 'với']

            tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords_list)
            sparse_matrix = tfidf_vectorizer.fit_transform(name_all_clean)
            index = [i for i in range(1, len(name_all_clean)+1)]
            doc_term_matrix = sparse_matrix.todense()
            df = pd.DataFrame(doc_term_matrix,
                              columns=tfidf_vectorizer.get_feature_names(),
                              index=index)
            feature_img = np.array(np.genfromtxt(
                "img_feature.txt", delimiter=","))
            df_ = pd.DataFrame(feature_img,
                               columns=[i for i in range(12672)],
                               index=[i for i in range(1, 49)])
            df_new = pd.concat([df, df_], axis=1)

            arr = rc_disease_similarity('static/image/' + img_path, model)
            # print(collection_treatment.find()[a])
            rc = ""
            r1 = ""
            r2 = ""
            res = ""
            rss = ""
            if arr is not None:
                for i in range(len(arr[0])):
                    # rc += "🌱" + arr[0][i] + "\n" + "🚑 " + arr[1][i] + "\n"
                    rc += arr[0][i] + '\n'
                    rss += arr[1][i] + '\n'

                # print(len(arr[0]), len(arr[1]))
                print(img_path, "Success")
                # rc = Markup(rc)
                # rc = rc.split('\n')
                return {
                    'name': name_all[a],
                    'symptom': symptom[a],
                    'treatment': treatment[a],
                    'rc': rc,
                    'rss':rss,
                    'predict': "desease"
                }
               
            else:
                print(img_path, "Healthy")
                return {'name': name_all[a], 'symptom': "", 'treatment': "", 'rc': "",'predict':'heatlthy' }
        else:
            print(img_path, "Failed")
            return ""
        
df_desc = pd.read_csv('./models/Data/Crop_Desc.csv', sep = ';', encoding = 'utf-8')
df = pd.read_csv('./models/Data/Crop_recommendation.csv')
rdf_clf = joblib.load('./models/crop_recommendation/RDF_model.pkl')

# rdf_clf = './models/crop_recommendation/RDF_model.pkl'
# rdf_clf = pickle.load(
#     open(rdf_clf, 'rb'))

X = df.drop('label', axis = 1)
y = df['label']

@app.route("/predict_crop",  methods=["GET", "POST"])
def predictcrop():
    global df_desc
    if request.method == 'POST':
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        ph = float(request.form['ph'])
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])
        rainfall = float(request.form['rainfall'])
        random_number = random.randint(1, 6)
        if random_number == 1:
            predict_inputs = [[N,P,K,temperature,humidity,ph,rainfall,1,0,0,0,0,0]]
        elif random_number == 2:
            predict_inputs = [[N,P,K,temperature,humidity,ph,rainfall,0,1,0,0,0,0]]
        elif random_number == 3:
            predict_inputs = [[N,P,K,temperature,humidity,ph,rainfall,0,0,1,0,0,0]]
        elif random_number == 4:
            predict_inputs = [[N,P,K,temperature,humidity,ph,rainfall,0,0,0,1,0,0]]
        elif random_number == 5:
            predict_inputs = [[N,P,K,temperature,humidity,ph,rainfall,0,1,0,0,1,0]]
        elif random_number == 6:
            predict_inputs = [[N,P,K,temperature,humidity,ph,rainfall,0,1,0,0,0,1]]

        rdf_predicted_value = rdf_clf.predict(predict_inputs)
        df_desc = df_desc.astype({'label':str,'image':str})
        df_desc['label'] = df_desc['label'].str.strip()
        df_desc['image'] = df_desc['image'].str.strip()
        df_pred_image = df_desc[df_desc['label'].isin(rdf_predicted_value)]
        df_image = df_pred_image['image'].item()

        # Parse the HTML
        soup = BeautifulSoup(df_image, 'html.parser')

        # Find the image tag
        img_tag = soup.find('img')

        # Get the src attribute value
        src_url = img_tag['src']
        plant_dict = {
            "rice": r"Gạo",
            "maize": r"Ngô",
            "chickpea": r"Đậu nành",
            "kidneybeans": r"Đậu tương",
            "pigeonpeas": r"Đậu xanh",
            "mothbeans": r"Đậu đen",
            "mungbean": r"Đậu xanh mung",
            "blackgram": r"Đậu đen",
            "lentil": r"Đậu lăng",
            "pomegranate": r"Lựu",
            "banana": r"Chuối",
            "mango": r"Xoài",
            "grapes": r"Nho",
            "watermelon": r"Dưa hấu",
            "muskmelon": r"Dưa lưới",
            "apple": r"Táo",
            "orange": r"Cam",
            "papaya": r"Đu đủ",
            "coconut": r"Dừa",
            "cotton": r"Bông",
            "jute": r"Đay",
            "coffee": r"Cà phê"
        }
    
        predict_res = plant_dict.get(rdf_predicted_value[0].encode('utf-8', 'ignore').decode('utf-8'))
        # print(plant_dict.get(rdf_predicted_value[0].encode('utf-8', 'ignore').decode('utf-8')))
        return  {'src_url': src_url, 'predict_res':predict_res}
    else:
        return {'src_url': "", 'predict_res':""}

@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('./models/Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        global db
        # Lấy dữ liệu từ biểu mẫu đăng ký
        username = request.form.get("name")
        password = request.form.get("password")
        email = request.form.get("email")
        collection_user = db['users']
                
        # Mã hóa mật khẩu
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        data = {
            'name': username,
            'email': email,
            'password': hashed_password.decode('utf-8'),
            'isAdmin': False,
            'verified': True,
            'followers': [],
            'following': [],
            'firstName': "",
            'lastName': "",
            'birthday': None,
            'phoneNumber': "",
            'gender': "",
            'createdAt': "",
            'updatedAt': "",
            'imgsrc': "https://static.vecteezy.com/system/resources/previews/019/896/008/original/male-user-avatar-icon-in-flat-design-style-person-signs-illustration-png.png",
            '__v': 0
        }

        # Lưu thông tin đăng ký vào cơ sở dữ liệu
        collection_user.insert_one(data)

        return render_template("index.html")

    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        global db
        collection_user = db['users']
        # Lấy thông tin đăng nhập từ biểu mẫu
        email = request.form.get("email")
        password = request.form.get("password")
        # Tìm người dùng trong MongoDB
        user = collection_user.find_one({"email": email})

        if user:
           # Kiểm tra mật khẩu
            hashed_password = user["password"].encode("utf-8")  # Convert the hashed password to bytes
            if bcrypt.checkpw(password.encode("utf-8"), hashed_password):
                # Đăng nhập thành công, lưu thông tin vào session
                session["username"] = user["name"]  
                session["imgsrc"] = user["imgsrc"]  
                return jsonify({"success": True, "message": "Login successful"})
        else:
            return jsonify({"success": False, "message": "Invalid email or password"})

@app.route("/logout")
def logout():
    session.clear()
    return render_template("index.html")


if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
