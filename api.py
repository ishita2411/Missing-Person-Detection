from flask import *
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2 
import face_recognition
import numpy as np
import smtplib


app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()
path = os.path.join(path, 'upload')
if not os.path.isdir(path):
    os.mkdir(path)

if not os.path.isdir(os.path.join(path, 'detected')):
    os.mkdir(os.path.join(path, 'detected'))

if not os.path.isdir(os.path.join(path, 'images')):
    os.mkdir(os.path.join(path, 'images'))

if not os.path.isdir(os.path.join(path, 'videos')):
    os.mkdir(os.path.join(path, 'videos'))

app.config['UPLOAD_FOLDER'] = path

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

names = []
embeddings = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_mail(name,loc):
    sender_email = "spoorthyss18@gmail.com"
    rec_email = "y.aaryan12@gmail.com"
    password = "spoorthy@123"
    message = "Detected " + name +' at ' + loc
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, rec_email, message)
    #print("mail sentt")

def encoding_file(img_name, person_name):
    img = face_recognition.load_image_file(os.path.join('upload', 'images', img_name))
    embedding = face_recognition.face_encodings(img)[0]
    embeddings.append(embedding)
    names.append(person_name)
    #print(names,embeddings)

def vid_detection(search_video):
    cap = cv2.VideoCapture(os.path.join('upload', 'videos', search_video))
    #cap = cv2.VideoCapture(0)

    if (cap.isOpened()== False): 
        print("Error opening video  file")

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
            face_loc = face_recognition.face_locations(frame)[0]
            #print('face_loc : ',face_loc)
            if face_loc:
                y1,x2,y2,x1 = face_loc

                img_new = frame[y1:y2,x1:x2]

                encodeTest = face_recognition.face_encodings(img_new)[0]

                results = face_recognition.compare_faces(embeddings, encodeTest)
                faceDis = face_recognition.face_distance(embeddings, encodeTest)
                #print(results,faceDis)

                match_index = np.argmin(faceDis)

                if results[match_index]:
                    name_found = names[match_index]
                    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'detected', (name_found + '.jpg')), img_new)
                    return name_found


                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break

    cap.release()
    cv2.destroyAllWindows()

    return None

@app.route("/", methods=["GET", "POST"])
def file():
    return render_template('file.html')


@app.route("/subm", methods=["GET", "POST"])
def subm():
    return render_template('upload.html')


@app.route("/multifile", methods=["GET", "POST"])
def multifile():
    files = request.files.getlist('files[]')

    dicti = request.form.to_dict()
    person_name = dicti['person_name']
    #print(person_name)

    i=1
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(person_name)
            filename += str(i)+'.' + file.filename.rsplit('.', 1)[1].lower()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],'images', filename))
            i+=1

            encoding_file(filename,person_name)

    flash('File(s) successfully uploaded')
    return redirect('/')


@app.route("/vinput", methods=["GET", "POST"])
def vinput():
    return render_template('videoupload.html')


@app.route("/videocheck", methods=["GET", "POST"])
def videocheck():
    dicti2 = request.form.to_dict()
    locations = dicti2['location']

    search_video = request.files.get('search_video') #to_dict()
    #print(search_video)
    search_video.save(os.path.join(app.config['UPLOAD_FOLDER'],'videos', (search_video.filename)))
    name_found = vid_detection((search_video.filename))
    #print('Found:',name_found)
    if not name_found:
        flash('No missing person recognized. Thank you.')
    else:
        send_mail(name_found,locations)
        flash('A missing person was recognized. Thank you.')

    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)