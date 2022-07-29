from flask import Flask, render_template, url_for, session, request, redirect, flash, jsonify
from flask_mysqldb import MySQL
import tensorflow as tf
import tensorflow_hub as hub
import MySQLdb.cursors
import os
from werkzeug.utils import secure_filename
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import joblib
import math
from math import sqrt

app=Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'EasyBuy'

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'mohan'

# Intialize MySQL
mysql = MySQL(app)

global embed
embed = hub.KerasLayer(os.getcwd())

class TensorVector(object):

    def __init__(self, FileName=None):
        self.FileName = FileName

    def process(self):
        img = tf.io.read_file(self.FileName)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
        features = embed(img)
        feature_set = np.squeeze(features)
        return list(feature_set)

def cosineSim(a1,a2):
    sum = 0
    suma1 = 0
    sumb1 = 0
    for i,j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        sum += i*j
    cosine_sim = sum / ((sqrt(suma1))*(sqrt(sumb1)))
    return cosine_sim

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/login')
def move_to_login():
    return render_template('login.html')

@app.route('/add_item')
def add_item():
    return render_template('add_item.html')

@app.route("/admin_dashboard", methods=['GET', 'POST'])
def login():
    info=''
    if request.method=='POST' and 'username' in request.form and 'password' in request.form:
        username=request.form['username']
        password=request.form['password']
        cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * from employee_details where username=%s and password=%s', (username, password, ))
        account=cursor.fetchone()
        if account:
            session['loggedin']=True
            session['username']=account['username']
            return render_template('add_item.html', username=session['username'])
        else:
            info='Incorrect username/password!'
    return render_template('login.html', info=info)

@app.route('/upload_item', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        brand=request.form['brand']
        item_name=request.form['item_name']
        category=request.form['category']
        item_category=request.form['item_category']
        item_category1=request.form['item_category1']
        price=request.form['price']
        desc=request.form['desc']
        cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("INSERT INTO items (`ID`, `Brand`,`item_name`,`img_name`,`description`,`Category`, `ItemCategory`,"
                       "`ItemCategory1`,`Price`)"
                       "  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",['',brand,item_name,filename,desc,category,item_category,item_category1,price])
        mysql.connection.commit()
        flash('Item added successfully!')
        return render_template('add_item.html')

@app.route('/')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/ImageSearch')
def move_to_ImageSearch():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM items ORDER BY ID ASC LIMIT 10")
    itemList = cur.fetchall()
    count = int(cur.rowcount)
    return render_template('image_search.html',itemList=itemList,count=count)

@app.route('/search')
def move_to_search():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT DISTINCT Category FROM items ORDER BY Category ASC")
    gender = cur.fetchall()  
    cur.execute("SELECT DISTINCT ItemCategory FROM items ORDER BY ItemCategory ASC")
    itemCategory=cur.fetchall()
    cur.execute("SELECT DISTINCT itemCategory1 FROM items ORDER BY itemCategory1 ASC")
    itemCategory1=cur.fetchall()
    return render_template('search.html', gender = gender, itemCategory=itemCategory, itemCategory1=itemCategory1)


@app.route('/analytics')
def move_to_analytics():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * from items ORDER BY ID DESC")
    item_list = cur.fetchall()  
    cur.execute("SELECT itemCategory1 as Category, COUNT(itemCategory1) as Count from items GROUP BY itemCategory1")
    itemCount1=cur.fetchall()
    data = {'Task' : 'Hours per Day', 'Work' : 11, 'Eat' : 2, 'Commute' : 2, 'Watching TV' : 2, 'Sleeping' : 7}
    print(data)
    print(type(data))
    itemCount1_dic={'Category' : 'Count'}
    for item in itemCount1:
        itemCount1_dic.update({item.get('Category'):item.get('Count')})
    print(itemCount1_dic)
    print(type(itemCount1_dic))
    cur.execute("SELECT ItemCategory, SUM(Price) as total From `items` GROUP BY ItemCategory")
    total_cost=cur.fetchall()
    print(total_cost)
    print(type(total_cost))
    item_total_price_dic={'Category': 'Price'}
    for item in total_cost:
        item_total_price_dic.update({item.get('ItemCategory'):item.get('total')})
    print(item_total_price_dic)
    return render_template('analytics.html',item_list=item_list, itemCount1_dic=itemCount1_dic,item_total_price_dic=item_total_price_dic)

class MultiHeadResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(MultiHeadResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            #print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            #print('Freezing intermediate layer parameters...')
        # change the final layers according to the number of categories
        self.l0 = nn.Linear(2048, 5) # for gender
        self.l1 = nn.Linear(2048, 7) # for masterCategory
        self.l2 = nn.Linear(2048, 45) # for subCategory
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2

device = torch.device('cpu')

model = MultiHeadResNet50(pretrained=False, requires_grad=False)
checkpoint = torch.load('models/model.pth',map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def model_predict(img_path, model):
    # read an image
    image = cv2.imread(img_path)
    #print(image)
    # keep a copy of the original image for OpenCV functions
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # apply image transforms
    image = transform(image)
    # add batch dimension
    image = image.unsqueeze(0).to(device)
    # forward pass the image through the model
    outputs = model(image)
    # extract the three output
    output1, output2, output3 = outputs
    # get the index positions of the highest label score
    out_label_1 = np.argmax(output1.detach().cpu())
    out_label_2 = np.argmax(output2.detach().cpu())
    out_label_3 = np.argmax(output3.detach().cpu())
    # load the label dictionaries
    num_list_gender = joblib.load('models/num_list_gender.pkl')
    num_list_master = joblib.load('models/num_list_master.pkl')
    num_list_sub = joblib.load('models/num_list_sub.pkl')
    # get the keys and values of each label dictionary
    gender_keys = list(num_list_gender.keys())
    gender_values = list(num_list_gender.values())
    master_keys = list(num_list_master.keys())
    master_values = list(num_list_master.values())
    sub_keys = list(num_list_sub.keys())
    sub_values = list(num_list_sub.values())
    final_labels = []
    # append the labels by mapping the index position to the values 
    final_labels.append(gender_keys[gender_values.index(out_label_1)])
    final_labels.append(master_keys[master_values.index(out_label_2)])
    final_labels.append(sub_keys[sub_values.index(out_label_3)])
    preds=final_labels[0]+" "+final_labels[1]+" "+final_labels[2]
    return preds


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'search_items', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
    return result

@app.route('/predict_image',methods=['GET','POST'])
def predict_image():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'search_items', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
    return result


def getEuclideanDistance(img1, img2):
    return math.sqrt(np.sum((img1-img2)**2))

@app.route("/fetch_records1",methods=["POST","GET"])
def fetch_similar_fashion_items():
    if request.method == 'POST':
        f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'search_items', secure_filename(f.filename))
    f.save(file_path)
    preds = model_predict(file_path, model)
    result = preds
    helper = TensorVector(file_path)
    vector = helper.process()
    arr= result.split()
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    search_text=arr[0]
    category=arr[1]
    category1=arr[2]
    cur.execute("SELECT * FROM items WHERE Category IN (%s) and ItemCategory IN (%s) and itemCategory1 IN(%s) ORDER BY ID ASC", [search_text,category,category1])
    itemlist = cur.fetchall()
    vectors=[]
    image_name=[]
    for item in itemlist:
        imagefilename = item.get('img_name')
        f = os.path.join(UPLOAD_FOLDER, imagefilename)
        helper = TensorVector(f)
        vector1=helper.process()
        simvalue = cosineSim(vector, vector1)
        vectors.append(simvalue)
        image_name.append(imagefilename)
    print(vectors)
    print(image_name)
    n=len(vectors)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if vectors[j] < vectors[j + 1] :
                vectors[j], vectors[j + 1] = vectors[j + 1], vectors[j]
                image_name[j], image_name[j + 1] = image_name[j+1], image_name[j]
    print(vectors)
    print(image_name)
    fashion_items=[]
    for item in image_name:
        cur.execute("SELECT * FROM items WHERE img_name IN(%s)", [item])
        record = cur.fetchall()
        fashion_items.append(record)
    return jsonify({'htmlresponse': render_template('response1.html', list=fashion_items,count=n)})


@app.route("/fetch_records",methods=["POST","GET"])
def fetch_similar_items():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'search_items', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        #print(file_path)
        random_image = cv2.imread(file_path,cv2.IMREAD_COLOR)
        img = cv2.resize(random_image, dsize=(224, 224))
        sample_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        predict_img=cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
        reimg=cv2.resize(predict_img,dsize=(96,96))
        Image_array=np.array(reimg)
        Image_data=Image_array.reshape(-1,96,96,1)
        Image_data=Image_data/255
        arr= result.split()
        similar_images=[]
        similar_images_names=[]
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        search_text=arr[0]
        category=arr[1]
        category1=arr[2]
        cur.execute("SELECT * FROM items WHERE Category IN (%s) and ItemCategory IN (%s) and itemCategory1 IN(%s) ORDER BY ID ASC", [search_text,category,category1])
        itemlist = cur.fetchall()
        for item in itemlist:
            imagefilename = item.get('img_name')
            f = os.path.join(UPLOAD_FOLDER, imagefilename)
            if os.path.isfile(f):
                random_image = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
            try:
                resized_img = cv2.resize(random_image, dsize=(96,96))
            except:
                print("Failed: ", f+".jpg")
            similar_images.append(resized_img)
            similar_images_names.append(imagefilename)
        #print(len(similar_images))
        #print(similar_images_names)
        re_image = np.array(similar_images).reshape(-1, 96,96,1)
        re_image = re_image/255
        distances=[]
        for i in range (0, len(re_image)):
            distances.append(getEuclideanDistance(Image_data,re_image[i]))
        sorted_distances = distances.copy()
        sorted_distances.sort()
        least_distances_sorted=sorted_distances[0:10]
        #print(distances)
        #print(least_distances_sorted)
        indexes=[]
        for i in range (0, len(least_distances_sorted)):
            indexes.append(distances.index(least_distances_sorted[i]))
        list=[]
        for i in indexes:
            list.append(similar_images_names[i])
        #print(list)
        fashion_items=[]
        for item in list:
            #print(item)
            cur.execute("SELECT * FROM items WHERE img_name IN(%s)", [item])
            record = cur.fetchall()
            fashion_items.append(record)
        #print(type(fashion_items))
        #print(fashion_items)
    return jsonify({'htmlresponse': render_template('response1.html', list=fashion_items,count=len(list))})

@app.route("/fetchrecords",methods=["POST","GET"])
def fetchrecords():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    numrows=0
    if request.method == 'POST':
        query = request.form['query']
        category = request.form['category']
        category1 = request.form['category1']
        if query == '' and category=='' and category1=='':
            cur.execute("SELECT * FROM items ORDER BY ID ASC LIMIT 10")
            itemlist = cur.fetchall()
            #print('all list')
        elif query !='' and category=='' and category1=='':
            search_text = request.form['query']
            #print(search_text)
            cur.execute("SELECT * FROM items WHERE Category IN (%s) ORDER BY ID ASC", [search_text])
            itemlist = cur.fetchall()
        elif query =='' and category!='' and category1=='':
            category = request.form['category']
            cur.execute("SELECT * FROM items WHERE ItemCategory IN (%s) ORDER BY ID ASC", [category])
            itemlist = cur.fetchall()
        elif query =='' and category=='' and category1!='':
            category1 = request.form['category1']
            cur.execute("SELECT * FROM items WHERE itemCategory1 IN (%s) ORDER BY ID ASC", [category1])
            itemlist = cur.fetchall()
        elif query !='' and category!='' and category1=='':
            search_text = request.form['query']
            category=request.form['category']
            #print(category)
            cur.execute("SELECT * FROM items WHERE Category IN (%s) and ItemCategory IN(%s) ORDER BY ID ASC", [search_text,category])
            itemlist = cur.fetchall()
        elif query !='' and category=='' and category1!='':
            search_text = request.form['query']
            category1=request.form['category1']
            cur.execute("SELECT * FROM items WHERE Category IN (%s) and itemCategory1 IN(%s) ORDER BY ID ASC", [search_text,category1])
            itemlist = cur.fetchall()
        elif query =='' and category!='' and category1!='':
            category = request.form['category']
            category1=request.form['category1']
            cur.execute("SELECT * FROM items WHERE ItemCategory IN (%s) and itemCategory1 IN(%s) ORDER BY ID ASC", [category,category1])
            itemlist = cur.fetchall()
        else:
            search_text=request.form['query']
            category = request.form['category']
            category1=request.form['category1']
            cur.execute("SELECT * FROM items WHERE Category IN (%s) and ItemCategory IN (%s) and itemCategory1 IN(%s) ORDER BY ID ASC", [search_text,category,category1])
            itemlist = cur.fetchall()
        numrows = int(cur.rowcount)
    return jsonify({'htmlresponse': render_template('response.html', itemlist=itemlist,numrows=numrows)})


if __name__ == "__main__":
    app.run(debug=True)