import os, glob
import face_recognition, argparse, shutil
import cv2, ntpath, numpy as np
from sklearn.cluster import DBSCAN
from imutils import build_montages

#ref (face[3], face[0]),(face[1], face[2]) - (x1,y1),(x2,y2)


ap=argparse.ArgumentParser()
ap.add_argument('-d','-dataset',required=True,help="folder which consist of images to be clustered")
ap.add_argument('-o','-outout_folder',required=True,help="folder to save the bounding box drawn images")
args=vars(ap.parse_args())
print(args)
input_folder_path=args['d']
des_path=args['o']
cluster_folder_path=input_folder_path+"_cluster_with_max_area_bbbox"
os.makedirs(des_path,exist_ok=True)

def generate_encodings_for_images(input_folder_path):
    input_images=glob.glob(os.path.join(input_folder_path,'*'))
    data_output=[]
    for image_path in input_images:
        image_dir_name,image_name=ntpath.split(image_path)
        image=cv2.imread(image_path)
        image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        bounding_boxes=face_recognition.face_locations(image_rgb,model='cnn') #cnn face detection model is more accurate and more run time
        bounding_boxes_area={}
        for bbox_no in range(len(bounding_boxes)): 
            bounding_boxes_area[(bounding_boxes[bbox_no][1]-bounding_boxes[bbox_no][3])*(bounding_boxes[bbox_no][2]-bounding_boxes[bbox_no][0])]=bounding_boxes[bbox_no]
        max_area_bbox_area=max(bounding_boxes_area.keys())
        max_area_bbox=bounding_boxes_area[max_area_bbox_area]
        face_encodings=face_recognition.face_encodings(image_rgb,[max_area_bbox])   #encodings, outputs the image in bgr format
        image_output=cv2.rectangle(image_rgb,(max_area_bbox[3],max_area_bbox[0]),(max_area_bbox[1],max_area_bbox[2]),color = (0, 255, 0),thickness = 2)
        image_output=cv2.cvtColor(image_output,cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(des_path,image_name),image_output)
        d = [{"ImagePath": image_name, "Bounding Box": box, "Encoding": enc} for (box, enc) in zip(bounding_boxes, face_encodings)]
        data_output.extend(d)
    return len(input_images),data_output

number_of_input_images,data_output=generate_encodings_for_images(input_folder_path)

data_output=np.array(data_output)
img_encodings=[value['Encoding'] for value in data_output]
print('number_of_input_images,len(img_encodings)',number_of_input_images,len(img_encodings))
dbscan_clustering_algo=DBSCAN(metric='euclidean',n_jobs=-1)
dbscan_clustering_algo.fit(img_encodings)
output_cluster_ids=np.unique(dbscan_clustering_algo.labels_)
print(output_cluster_ids)
print(len(output_cluster_ids))
numUniqueFaces = len(np.where(output_cluster_ids > -1)[0])  #taking outlier(image is not near to any other image) out
print(numUniqueFaces)
print('number of unique faces:',numUniqueFaces)


def img_read_resize(input_folder_path,data_output,index):
    image_name=data_output[index]['ImagePath']
    image=cv2.imread(os.path.join(input_folder_path,image_name))
    (top, right, bottom, left) = data_output[index]["Bounding Box"]
    face = image[top:bottom, left:right]
    face = cv2.resize(face, (96, 96)) 
    return image_name,face

for cluster_id in output_cluster_ids:
    os.makedirs(os.path.join(cluster_folder_path,str(cluster_id)),exist_ok=True)
    indexes=np.where(dbscan_clustering_algo.labels_== cluster_id)[0]
    indexes = np.random.choice(indexes, size=min(25, len(indexes)),replace=False)
    faces=[]
    print("********Cluster number*********",cluster_id)
    for i in indexes:
        image_name,face=img_read_resize(input_folder_path,data_output,i)
        faces.append(face)
        print(image_name)
        shutil.copy(os.path.join(input_folder_path,image_name),os.path.join(cluster_folder_path,str(cluster_id),image_name))
    montage = build_montages(faces, (84, 84), (5, 5))[0]
    title = "Face ID #{}".format(cluster_id)
    title = "Faces that is not near to another face" if cluster_id == -1 else title
    cv2.imshow(title, montage)
    cv2.waitKey(0)








