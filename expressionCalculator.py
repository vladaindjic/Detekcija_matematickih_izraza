"""
    Sve slike pravljenje koriscenjem Pages aplikacija na MacOS, pri zoomu od 175%
    i velicine fonta od 24pt i 75pt. Na taj nacin mozemo da prepoznamo izraze
    i na krupnim i na sitnim slikama.

    Korisceni fontovi su:
        - helvetica
        - times
        - arial
        - times new roman
        - PT Mono
        - America Typerwriter
        - Apple Chancery
        - Comic Sans MS
        - Verdana
        - Andale Mono


"""


import os,cv2, sys, time, shutil
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
# iako TensorFlow radi u pozadini, koristicemo notaciju Theana
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model



"""
================================================================================
    Deo koji sluzi za pripremu dataseta
"""


MIN_CONTOUR_AREA = 50

RESIZED_IMAGE_WIDTH = 32
RESIZED_IMAGE_HEIGHT = 32


CURRENT_PATH = os.getcwd()
INPUT_FOLDER_PATH = CURRENT_PATH + "/inputImages"
OUTPUT_FOLDER_PATH = CURRENT_PATH + "/outputImages"

class MyResizedImage(object):
    """
        Klasa koja sadrzi konturu znaka i njegovu sliku dimenzija 32x32.
    """
    def __init__(self, X, Y, width, height, image):
        self.X = X
        self.Y = Y
        self.width = width
        self.height = height
        self.image = image


def makeBlackAndWhitePhoto(path):
    """
        Od fotografije koja se nalazi na prosledjenoj putanji,
        pravimo crno-belu sliku
    """

    imgTrainingNumbers = cv2.imread(path)                       # ucitavanje slike

    if imgTrainingNumbers is None:                              # ako slika nije ucitana
        print ("error: Slika nije ucitana iz fajla: {0} \n\n".format(path))          # prikazujemo gresku
        sys.exit()                                              # i gasimo program
        return

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # dobije grayscale slike
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blurujemo

    # ocistimo sliku da bude crno bela
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # ulazna slika
                                      255,                                  # pixeli koji prodju threshold neka budu beli
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # gausian daje bolje rezultate od meana
                                      cv2.THRESH_BINARY_INV,                # pozadina da bude crna, a slova da budu bela
                                      41,                                   # velicina piksela suseda koji se koristi za racunanje threshold
                                      2)                                    # konstanta koja se oduzima od sredine
    # vraticemo sliku koja je potpuno crno bela
    # cv2.imshow("Ovako izgleda kada thresholdujemo", imgThresh)
    # cv2.waitKey(0)
    return imgThresh



def getImageOfCharacters(photo):
    """
        Metoda koja sa prosledjene izdvaja konture u kojima se nalazi tekst.
        Sortira ih po X koordinati, kako bi bile poredjane onako kako se
        pojavljuju na slici.
    """
    # kopija slike
    copyPhoto = photo.copy();

    # izdvajamo konture
    imgContours, npaContours, npaHierarchy = cv2.findContours(copyPhoto,        # Korisitmo kopiju slike
                                                 cv2.RETR_EXTERNAL,                 # izmemo samo onu konturu koja je najvise spolja
                                                 cv2.CHAIN_APPROX_SIMPLE)           # kompresujemo horizontalne, vertikalne i dijagonalne segmente i ostavljamo samo njihove krajnje tacke


    # cv2.imshow("Moze li ovako? ", photo)
    # cv2.waitKey(0)

    resized_images = []

    # prolazimo kroz sve konture
    for npaContour in npaContours:
        # provera da li je kontura dovoljno velika
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:

            # izvlacimo njen pravougaonik
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)


            # # # # ukoliko zelimo da crtamo konture, moze se ukljuciti za debagovanje
            # cv2.rectangle(photo,           # draw rectangle on original training image
            #               (intX, intY),                 # gornji levi ugao
            #               (intX+intW,intY+intH),        # donji desni ugao
            #               (255, 255, 255),                  # bela kontura
            #               2)                            # debljina konture

            imgROI = photo[intY:intY+intH, intX:intX+intW]        # isecemo sa glavne slike                           # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # odradimo resizeovanje
            resized_images.append(MyResizedImage(intX, intY, intW, intH, imgROIResized))



            # cv2.imshow("Kada se resizuje", imgROIResized)
            # cv2.imshow("original", photo)
            # cv2.waitKey(0)

    resized_images.sort(key=lambda i: i.X, reverse=False)
    return resized_images



def prepareCharacters():
    """
        U direktorijumu inputImages pronalazimo sve datoteke i jednu po jednu
        prosledjujemo metodi prepareOneCharacter
    """
    # treba pripremiti odgovarajucu hijerarhiju direktorijuma
    if not os.path.isdir(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    for path in os.listdir(INPUT_FOLDER_PATH):
        if path.endswith(".png"):
            prepareOneCharacter(INPUT_FOLDER_PATH + "/" + path)

    cv2.destroyAllWindows();


def prepareOneCharacter(path):
    """
        Metoda koja na osnovu prosledjene putanje, ucitava sliku i iz nje
        izdvaja sve fontove jednog karaktera i smeta na odgovarajucu putanju.
    """

    charName = path.split("/")[-1].split(".")[0]

    # folder za karakter
    characterFolderPath = OUTPUT_FOLDER_PATH + '/' + charName
    if not os.path.isdir(characterFolderPath):
        os.makedirs(characterFolderPath)


    # print("Ovo je putanja: {0}, a ovo su karakteri: {1}".format(path, charName))
    blackAndWhitePhoto = makeBlackAndWhitePhoto(path)
    countours = getImageOfCharacters(blackAndWhitePhoto)
    if(charName == "assignment"):
        prepareAssignment(blackAndWhitePhoto, countours, OUTPUT_FOLDER_PATH + '/assignment')
    else:
        prepareOtherCharacter(blackAndWhitePhoto, countours, OUTPUT_FOLDER_PATH + '/' + charName, charName)


def prepareOtherCharacter(photo, myResizedImages, characterFolderPath, charName):
    for i, myResizedImage in enumerate(myResizedImages):
        cv2.imwrite("{0}/{1}_{2}.png".format(characterFolderPath, charName, i), myResizedImage.image)


def prepareAssignment(photo, rectangles, characterFolderPath):
    """
        Ne bas pametan nacin za obradu znaka jednakosti.
        Najveci problem je sto znak = prepoznaje kao minus.
        Vec smo odredili pravougaonike, treba da vidimo koji se poklapaju
        i da ih izdvojimo.
        Moramo posebno izdvojiti, jer se unutar znaka jednako nalazi praznina
    """
    # sada treba da sortiramo pravougaonike po x koordinati
    rectangles.sort(key=lambda r: r.X, reverse=False)

    charName = "assignment"
    numOfChars = 0
    for i in range(0, len(rectangles), 2):
        # uzmemo par pravougaonika
        firstRec = rectangles[i]
        secondRec = rectangles[i+1]

        # gornja leva tacka od koje se krece
        leftX = min(firstRec.X, secondRec.X)
        leftY = min(firstRec.Y, secondRec.Y)
        # donja desna tacka
        rightX = max(firstRec.X + firstRec.width, secondRec.X + secondRec.width)
        rightY = max(firstRec.Y + firstRec.height, secondRec.Y, secondRec.height)


        # crop i resize
        # imgROI = photo[intY:intY+intH, intX:intX+intW]        # isecemo sa glavne slike                           # crop char out of threshold image

        imgROI = photo[leftY : rightY,  leftX : rightX]       # isecemo sa glavne slike                           # crop char out of threshold image


        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # odradimo resizeovanje
        # pa stampanje u datoteku
        numOfChars += 1
        cv2.imwrite("{0}/{1}_{2}.png".format(characterFolderPath, charName, str(numOfChars)), imgROIResized)

        # cv2.rectangle(photo,           # draw rectangle on original training image
        #               (leftX, leftY),                 # gornji levi ugao
        #               (rightX, rightY),        # donji desni ugao
        #               (255, 255, 255),                  # bela kontura
        #               2)
        # cv2.imshow("jednacici moji slatki", photo)
        # cv2.waitKey(0)



"""
================================================================================
    Deo za Simple Convolution Neural Network
"""

NUMBER_OF_CHARACTERS = 16  # trenutno necemo obraditi znak minus
NUMBER_OF_CHANNELS = 1  # samo crno bele slike gledamo


img_rows=RESIZED_IMAGE_WIDTH
img_cols=RESIZED_IMAGE_HEIGHT
num_channel=NUMBER_OF_CHANNELS
num_epoch=10  # pokazalo se da jako dobro radi za 10 epoha

# Broj klasa koje koristimo za nadgledano ucenje
num_classes = NUMBER_OF_CHARACTERS
# Znaci u redosledu kako ih ucitavamo
names = ['0','1','2','3', '4','5','6','7','8','9', '/', '(', '-', '*', '+', ')']

# '=', '/', '(', '-', '*', '+', ')']


def loadDatasetInProperFormat():
    """
        Metoda koja ucitava pripremljen dataset u odgovrajucem formatu koji
        odgovara Kerasovoj CNN.
    """
    # Definisemo putanje sa koje citamo
    data_path = CURRENT_PATH + '/outputImages'
    # izlistavamo poddirektorijume
    data_dir_list = os.listdir(data_path)

    img_data_list=[]

    print("\n\n\n")
    for dataset in data_dir_list:

    	dir_path = data_path + '/' + dataset

    	if os.path.isfile(dir_path):
    		#print("Imamo jedan fajl")  # ako se slucajno nadje neki fajl, preskacemo ga
    		continue
        # Izlistavanje fotografija
    	img_list=os.listdir(dir_path)
    	for img in img_list:
            # ucitavanje slike, prebacivanje u GrayScale, resize na 32x32
    		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
    		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    		input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
    		img_data_list.append(input_img_resize)


    print("\n\n\n\tVelicina dataseta je: {0}".format(len(img_data_list)))
    # slike iz liste prebacujemo u numpy niz
    img_data = np.array(img_data_list)
    # prebacujemo u float32 reprezentaciju
    img_data = img_data.astype('float32')
    # vrsimo normalizaciju
    img_data /= 255
    print ("\n\n\n\tOvako nam izgleda dataset kada se ucita u numpy niz: {0}".format(img_data.shape))


    # posto radimo sa Theanom
    img_data= np.expand_dims(img_data, axis=1)
    print (img_data.shape)
    return img_data



def makeLabels(img_data):
    """
        Vrsimo labeliranje ucitanih podataka.
    """

    num_of_samples = img_data.shape[0]
    num_of_samples_per_class = num_of_samples // NUMBER_OF_CHARACTERS
    print("\n\n\n\tOvoliko imamo primeraka po klasi: {0}".format( num_of_samples_per_class))
    labels = np.ones((num_of_samples,),dtype='int64')

    for i in range(NUMBER_OF_CHARACTERS):
        labels[i*num_of_samples_per_class : (i+1)*num_of_samples_per_class] = i

    return labels


def simpleCNNModel(img_data):
    """
        Kreiramo jednostavnu konvolucijsku mrezu.
    """
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=img_data[0].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    num_pixels = RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # kompajliramo model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def makeSimpleCNN():
    """
        Metoda koja:
            1. ucitava podatke u odgovarajucem formatu
            2. vrsi njihovo labeliranje
            3. prebacuje labele u izlazni format mreze (niz od 16 koji ima samo
                jednu jedinicu)
            4. Malo promesamo podatke
            5. Odvojimo testne podatke, da budu 1/5 dataseta
            6. Kreiramo model
            7. Fitujemo model
            8. Vrsimo procenu modela i prikazujemo gresku
    """

    img_data = loadDatasetInProperFormat()
    labels = makeLabels(img_data)
    # labele u odgovarajucem formatu
    Y = np_utils.to_categorical(labels, num_classes)
    # Malo promesamo podatke
    x,y = shuffle(img_data,Y, random_state=2)
    # Podelimo dataset na training_set i test_set u odnosu 4:1
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    # Definisemo model
    model = simpleCNNModel(img_data)
    startTime = time.time()
    # Radimo treniranje
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, batch_size=10, verbose=2)
    endTime = time.time()
    timeNeedForTraining = endTime - startTime
    logCNN("\n>>>Vreme potrebno za treniranje CNN je: {0}ms.".format(timeNeedForTraining))
    # prikazujemo informacije o samom procesu treniranja
    showInformationAboutTraining(hist, num_epoch)
    # Procena modela
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("\n\n\n\tOvo je greska nakon treniranja jednostavne CNN: {0:.2f}%".format(100-scores[1]*100))
    logCNN("\n>>>Ovo je greska nakon treniranja jednostavne CNN: {0:.2f}%".format(100-scores[1]*100))
    return model




"""
================================================================================
    Deo koji sluzi za prepoznavanja izraza i njegovo racunanje
"""

PATH_EXPRESSIONS = CURRENT_PATH + "/expressions"
PATH_RESULTS = CURRENT_PATH + "/results"
PATH_ERRORS = CURRENT_PATH + "/errors"
PATH_LOGS = CURRENT_PATH + "/logs"
PATH_RESULTS_SIMPLE_CNN = PATH_RESULTS + "/simple_cnn.txt"
PATH_RESULTS_SIMPLE_NN = PATH_RESULTS + "/simple_nn.txt"
PATH_RESULTS_REAL= PATH_RESULTS + "/real.txt"
PATH_ERRORS_NN = PATH_ERRORS + "/nn.txt"
PATH_ERRORS_CNN = PATH_ERRORS + "/cnn.txt"
PATH_LOGS_NN = PATH_LOGS + "/nn.txt"
PATH_LOGS_CNN = PATH_LOGS + "/cnn.txt"


def findPartsOfAllExpressions():
    """
        Funkcija koja nadje sve delove svih izraza
    """
    # ovde cemo cuvati sve regione svih izraza
    expressions_parts = []
    expressions_dir_list = os.listdir(PATH_EXPRESSIONS)
    for exp_path in expressions_dir_list:
        # regioni jednog izraza
        if not exp_path.endswith(".png"):
            continue
        expression_parts = findPartsOfOneExpression(PATH_EXPRESSIONS+"/"+exp_path)
        expressions_parts.append(expression_parts)
    return expressions_parts

def findPartsOfOneExpression(image_path):
    """
        Funkcija koja trazi delove jednog izraza

        Return:
            list<MyResizedImage>
    """
    # prvo sliku prebacimo u crno belo
    blackAndWhitePhoto = makeBlackAndWhitePhoto(image_path)
    # izdvojimo objekte klase MyResizedImage
    resized_images = getImageOfCharacters(blackAndWhitePhoto)
    # for i in resized_images:
    #     cv2.imshow("slicica", i.image)
    #     cv2.waitKey(0)
    # vratimo slike
    return resized_images



def calucateAllExpressionsSimpleCNN(exrepssions_parts, model):
    """
        Funkcija koja racuna sve vrednosti izraza koriscenjem CNN
    """
    output = ""
    for i, expression_parts in enumerate(exrepssions_parts):
        expressionAndResult = calucateOneExpressionSimpleCNN(expression_parts, model)
        output += expressionAndResult + "\n"
        print("Izraz sa rednim brojem {0}. je: {1}.".format(i, expressionAndResult))
    # upisujemo rezultate u odgovarajuci fajl
    f = open(PATH_RESULTS_SIMPLE_CNN, 'w')
    f.write(output.strip())
    f.close()


def calucateOneExpressionSimpleCNN(expression_parts, model):
    """
        Metoda koja slike pronadjenih karaktera pusta na CNN
        i vraca string oblika: "string_izraz=vrednost_izraza".
    """
    # predikovani karakteri
    prediction_characters = []
    # prolazimo kroz sve delove izraza koji su objekti klase MyResizedImage
    for myResizedImage in expression_parts:
        # pripremimo sliku za predikciju
        predict_image = prepareImageForPredictionCNN(myResizedImage)
        predited_class = model.predict_classes(predict_image)
        prediction_characters.append(names[predited_class[0]])

    # kako izgleda prepoznati izra
    string_expression = "".join(prediction_characters)
    # koja je njegova vrednost
    print("Ovde mi nesto puca a nemam pojma zasto: {0}".format(string_expression))
    try:
        # racunamo vrednost
        result_expression = eval(string_expression)
        # string koji vracamo ima format "string_izra=vrednost_izraza"
        retVal = "{0}={1}".format(string_expression, result_expression)
        return retVal
    except:
        # ako ne moze da se izracuna, samo vratimo sta smo prepoznali
        return string_expression


def prepareImageForPredictionCNN(myResizedImage):
    """
        Metoda koja od objekta klase MyResizedImage pravi odgovarajucu sliku
        za pustanje u neuralnu mrezu
    """
    test_image = myResizedImage.image
    # test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # test_image=cv2.resize(test_image,(RESIZED_IMAGE_WIDTH,RESIZED_IMAGE_HEIGHT))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255

    print (test_image.shape)

    # imamo 1 kanal i radimo sa Theanom
    test_image= np.expand_dims(test_image, axis=0)
    test_image= np.expand_dims(test_image, axis=0)
    print (test_image.shape)
    return test_image





"""
================================================================================
    Deo za validaciju.
"""

def readRealResults():
    """
        Funkcija koja ucitava prave rezultate
    """

    real_results = []
    # ucitavamo prave rezultate
    f = open(PATH_RESULTS + "/real.txt", 'r')
    real_results = f.read().strip().split("\n");
    f.close()
    print("\n\n\n\tUkupan broj izraza je: {0}.".format(len(real_results)))
    return real_results


def evaluateSimpleCNN(real_results):
    """
        Funkcija koja ucitava rezultate koje je jednostavna CNN ucitala,
        poredi ih sa pravim rezultati i vraca procenta pogadjanja.
    """
    f = open(PATH_RESULTS_SIMPLE_CNN)
    simple_cnn_results = f.read().strip().split("\n")
    f.close()
    number_of_expressions = len(real_results)
    if number_of_expressions != len(simple_cnn_results):
        raise Exception("Nisu svi izrazi izracunati")

    # broj izraza koji se poklapaju
    number_of_matched = 0
    for i in range(number_of_expressions):
        if real_results[i].strip() == simple_cnn_results[i].strip():
            number_of_matched += 1
        else:
            errorCNN(real_results[i], simple_cnn_results[i], i)

    # procentualna uspesno jednostavne CNN
    procent = (number_of_matched / number_of_expressions) * 100
    print("\n\n\n\tBroj tacno izracunatih izraza koriscenjem jednostavno CNN je: {0}"\
        "\n\tUspesnost postignuta primenom jednostavne CNN je: {1}%."\
            .format(number_of_matched, procent))
    logCNN("\n>>>Broj tacno izracunatih izraza koriscenjem jednostavno CNN je: {0}"\
           "\n\tUspesnost postignuta primenom jednostavne CNN je: {1}%."\
            .format(number_of_matched, procent))

    return procent




"""
================================================================================
    Deo za jednostavnu NN
"""

def makeSimpleNN():
    """
        Metoda koja:
            1. ucitava podatke u odgovarajucem formatu
            2. vrsi njihovo labeliranje
            3. prebacuje labele u izlazni format mreze (niz od 16 koji ima samo
                jednu jedinicu)
            4. Malo promesamo podatke
            5. Odvojimo testne podatke, da budu 1/5 dataseta
            6. Kreiramo model
            7. Fitujemo model
            8. Vrsimo procenu modela i prikazujemo gresku
    """

    img_data = loadDatasetInProperFormat()
    labels = makeLabels(img_data)
    # labele u odgovarajucem formatu
    Y = np_utils.to_categorical(labels, num_classes)
    # Malo promesamo podatke
    x,y = shuffle(img_data,Y, random_state=2)
    # Podelimo dataset na training_set i test_set u odnosu 4:1
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # -------------------------------------------------

    # sliku dimenzije 32x32 prebacujemo i niz od 1024 pixela
    # kod Theana je (broj_primeraka, broj kanala, sirina, visina)
    num_pixels = X_train.shape[2] * X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

    # Definisemo model
    model = simpleNNModel(img_data)
    startTime = time.time()
    # Radimo trenisajne
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, batch_size=200, verbose=2)
    endTime = time.time()
    timeNeedForTraining = endTime - startTime
    logNN("\n>>>Vreme potrebno za treniranje NN: {0}ms.".format(timeNeedForTraining))
    # prikazujemo informacije o samom procesu treniranja
    showInformationAboutTraining(hist, num_epoch)
    # Procena modela
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("\n\n\n\tOvo je greska nakon treniranja jednostavne NN: {0:.2f}%".format(100-scores[1]*100))
    logNN("\n>>>Ovo je greska nakon treniranja jednostavne NN: {0:.2f}%".format(100-scores[1]*100))
    return model



def simpleNNModel(img_data):
    """
        Kreiramo jednostavnu neuralnu mrezu.
    """

    # da uvek dobijemo isti rezultat
    seed = 7
    np.random.seed(seed)

    # Kreiramo model
    num_pixels = RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # kompajliramo ga
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def calucateAllExpressionsSimpleNN(exrepssions_parts, model):
    """
        Funkcija koja racuna sve vrednosti izraza koriscenjem NN
    """
    output = ""
    for i, expression_parts in enumerate(exrepssions_parts):
        expressionAndResult = calucateOneExpressionSimpleNN(expression_parts, model)
        output += expressionAndResult + "\n"
        print("\n\n\n\tIzraz sa rednim brojem {0}. je: {1}.".format(i, expressionAndResult))
    # upisujemo rezultate u odgovarajuci fajl
    f = open(PATH_RESULTS_SIMPLE_NN, 'w')
    f.write(output.strip())
    f.close()

def calucateOneExpressionSimpleNN(expression_parts, model):
    """
        Metoda koja slike pronadjenih karaktera pusta na NN
        i vraca string oblika: "string_izraz=vrednost_izraza".
    """
    # predikovani karakteri
    prediction_characters = []
    # prolazimo kroz sve delove izraza koji su objekti klase MyResizedImage
    for myResizedImage in expression_parts:
        # pripremimo sliku za predikciju
        predict_image = prepareImageForPredictionNN(myResizedImage)
        predited_class = model.predict_classes(predict_image)
        prediction_characters.append(names[predited_class[0]])

    # kako izgleda prepoznati izra
    string_expression = "".join(prediction_characters)
    try:
        # koja je njegova vrednost
        result_expression = eval(string_expression)
        # string koji vracamo ima format "string_izra=vrednost_izraza"
        retVal = "{0}={1}".format(string_expression, result_expression)
        return retVal
    except:
        # vratimo koja je greska
        return string_expression


def prepareImageForPredictionNN(myResizedImage):
    """
        Metoda koja od objekta klase MyResizedImage pravi odgovarajucu sliku
        za pustanje u neuralnu mrezu
    """
    test_image = myResizedImage.image
    # test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # test_image=cv2.resize(test_image,(RESIZED_IMAGE_WIDTH,RESIZED_IMAGE_HEIGHT))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255

    print (test_image.shape)

    # imamo 1 kanal i radimo sa Theanom
    test_image= np.expand_dims(test_image, axis=0)
    test_image= np.expand_dims(test_image, axis=0)
    print (test_image.shape)


    # sada jos sliku iz matricnog oblika da prebacimo u niz
    num_pixels = test_image.shape[2] * test_image.shape[3]
    test_image = test_image.reshape(test_image.shape[0], num_pixels).astype('float32')

    return test_image



def evaluateSimpleNN(real_results):
    """
        Funkcija koja ucitava rezultate koje je jednostavna CNN ucitala,
        poredi ih sa pravim rezultati i vraca procenta pogadjanja.
    """
    f = open(PATH_RESULTS_SIMPLE_NN)
    simple_nn_results = f.read().strip().split("\n")
    f.close()
    number_of_expressions = len(real_results)
    if number_of_expressions != len(simple_nn_results):
        raise Exception("Nisu svi izrazi izracunati")

    # broj izraza koji se poklapaju
    number_of_matched = 0
    for i in range(number_of_expressions):
        if real_results[i].strip() == simple_nn_results[i].strip():
            number_of_matched += 1
        else:
            errorNN(real_results[i], simple_nn_results[i], i)

    # procentualna uspesno jednostavne NN
    procent = (number_of_matched / number_of_expressions) * 100
    print("\n\n\n\tBroj tacno izracunatih izraza koriscenjem jednostavno NN je: {0}"\
        "\n\tUspesnost postignuta primenom jednostavne NN je: {1}%."\
            .format(number_of_matched, procent))
    logNN("\n>>>Broj tacno izracunatih izraza koriscenjem jednostavno NN je: {0}"\
        "\n\tUspesnost postignuta primenom jednostavne NN je: {1}%."\
            .format(number_of_matched, procent))

    return procent





"""
================================================================================
    Globalni deo
"""

def errorNN(realExpression, predictedExpression, number):
    f = open(PATH_ERRORS_NN, "a")
    output = "==========\n"
    output += "Expresion: {0}.\n".format(number)
    output += "Real expression: {0}\n".format(realExpression)
    output += "Predicted value: {0}\n".format(predictedExpression)
    f.write(output)
    f.close()

def logNN(output):
    f = open(PATH_LOGS_NN, "a")
    f.write(output)
    f.close()

def errorCNN(realExpression, predictedExpression, number):
    f = open(PATH_ERRORS_CNN, "a")
    output = "==========\n"
    output += "Expresion: {0}.\n".format(number)
    output += "Real expression: {0}\n".format(realExpression)
    output += "Predicted value: {0}\n".format(predictedExpression)
    f.write(output)
    f.close()

def logCNN(output):
    f = open(PATH_LOGS_CNN, "a")
    f.write(output)
    f.close()


def showInformationAboutTraining(hist, num_epoch):
    """
        Funkcija koja prikazuje informacije o procesu treniranja
    """

    # prikazujemo gubitak i preciznost
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(num_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.show()
    input("")





def processSimpleCNN():
    """
        Metoda koja obavlja ceo postupak jednostavne CNN:
            1. izgradnju CNN
            2. izdvajanje karaktera sa slike
            3. ucitavanje stvarnih rezultata
            4. provera koliko dobro CNN radi
    """

    simpleCNNModel = None
    # da li je potrebno treniranje jednostavne CNN
    if not os.path.isfile(CURRENT_PATH+"/simpleCNNModel.hdf5"):
        print("\n\n\n\t***Proces treniranja CNN.")
        simpleCNNModel = makeSimpleCNN()
        simpleCNNModel.save("simpleCNNModel.hdf5")
    else:
        simpleCNNModel = load_model("simpleCNNModel.hdf5")

    # jednostavna CNN mreza trazi i racuna izraze
    exrepssions_parts = findPartsOfAllExpressions()
    # racunamo sve izraze koriscenjem CNN
    calucateAllExpressionsSimpleCNN(exrepssions_parts, simpleCNNModel)
    # ucitavamo sve izraze
    real_results = readRealResults()
    # provera koliko dobro radi CNN
    evaluateSimpleCNN(real_results)


def processSimpleNN():
    """
        Metoda koja obavlja ceo postupak jednostavne CNN:
            1. izgradnju NN
            2. izdvajanje karaktera sa slike
            3. ucitavanje stvarnih rezultata
            4. provera koliko dobro NN radi
    """

    simpleNNModel = None
    # da li je potrebno treniranje jednostavne CNN
    if not os.path.isfile(CURRENT_PATH+"/simpleNNModel.hdf5"):
        print("\n\n\n\t***Proces treniranja NN.")
        simpleNNModel = makeSimpleNN()
        simpleNNModel.save("simpleNNModel.hdf5")
    else:
        simpleNNModel = load_model("simpleNNModel.hdf5")

    # jednostavna CNN mreza trazi i racuna izraze
    exrepssions_parts = findPartsOfAllExpressions()
    # racunamo sve izraze koriscenjem NN
    calucateAllExpressionsSimpleNN(exrepssions_parts, simpleNNModel)
    # ucitavamo sve izraze
    real_results = readRealResults()
    # provera koliko dobro radi CNN
    evaluateSimpleNN(real_results)


def clearProject():
    """
        Metoda koja uklanja datset i cisti foldere logs i errors.
    """
    # uklanjamo dataset
    if os.path.isdir(OUTPUT_FOLDER_PATH):
        shutil.rmtree(OUTPUT_FOLDER_PATH)

    # uklanjamo sve iz logs foldera
    if os.path.isfile(PATH_LOGS_NN):
        os.remove(PATH_LOGS_NN)
    if os.path.isfile(PATH_LOGS_CNN):
        os.remove(PATH_LOGS_CNN)

    #uklanjamo sve iz error foldera
    if os.path.isfile(PATH_ERRORS_NN):
        os.remove(PATH_ERRORS_NN)
    if os.path.isfile(PATH_ERRORS_CNN):
        os.remove(PATH_ERRORS_CNN)

    # cistimo trenirane mreze
    if os.path.isfile(CURRENT_PATH + "/simpleNNModel.hdf5"):
        os.remove(CURRENT_PATH + "/simpleNNModel.hdf5")
    if os.path.isfile(CURRENT_PATH + "/simpleCNNModel.hdf5"):
        os.remove(CURRENT_PATH + "/simpleCNNModel.hdf5")


def main():
    """
        Funkcija od koje krece izvrsavanje programa.
    """
    # da li treba sve da se ocisit
    if len(sys.argv) == 2:
        if sys.argv[1] == "clear":
            clearProject()


    # da li je potrebna priprema dataseta
    if not os.path.isdir(OUTPUT_FOLDER_PATH):
        print("\n\n\n\t***Pripremamo dataset.")
        prepareCharacters()

    processSimpleCNN()
    processSimpleNN()








if __name__ == '__main__':
    main()
