# importing Qt widgets
from PyQt5.QtWidgets import *
 
# importing system
import sys
 
# importing numpy as np
import numpy as np
 
# importing pyqtgraph as pg
import pyqtgraph as pg
from pyqtgraph.opengl import GLScatterPlotItem, GLViewWidget
from pyqtgraph.opengl.items.GLAxisItem import GLAxisItem
from pyqtgraph.opengl.items.GLGridItem import GLGridItem
pg.setConfigOptions(useOpenGL=True)
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSignal
import sounddevice as sd
import soundfile as sf
import copy
import umap
import librosa
import glob
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors

class PickableGLViewWidget(GLViewWidget):
    pickedPoint = pyqtSignal(int)
    def __init__(self, pos, parent=None, devicePixelRatio=None, rotationMethod='euler', picking_radius=10):
        super().__init__(parent=parent, devicePixelRatio=devicePixelRatio, rotationMethod=rotationMethod)
        self.pos = pos
        self.picking_radius = picking_radius
        self.text = ''
    
    def mouseReleaseEvent(self, ev):
        view_matrix = self.projectionMatrix()*self.viewMatrix()
        inv_view_matrix = np.array(view_matrix.data()).reshape(4,4)
        wpos = np.concatenate([self.pos,np.ones((self.pos.shape[0],1))],axis=1)
        camera_space_pos = np.matmul(wpos,inv_view_matrix)
        camera_space_pos = camera_space_pos[:,:2]/camera_space_pos[:,2,np.newaxis]
        canvas_size = self.size()
        pixel_space_pos = ((camera_space_pos + 1)/2)*np.array([canvas_size.width(),canvas_size.height()])
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pixel_space_pos)
        mouse_pos = ev.pos()
        distances, indices = nbrs.kneighbors(np.array([[mouse_pos.x(),canvas_size.height()-mouse_pos.y()]]))
        if distances[0][0] < self.picking_radius:
            selected_idx = indices[0][0]
        else:
            selected_idx = None        
        self.pickedPoint.emit(selected_idx)

        if selected_idx is None:
            return super().mouseReleaseEvent(ev)

    def paintGL(self, region=None, viewport=None, useItemNames=False):
        super().paintGL(region=region, viewport=viewport, useItemNames=useItemNames)

    def set_text(self,text):
        self.text = text

class Window(QMainWindow):
    def __init__(self, x, y, z=None,c=None, s=None, audios=None, normalize=True):
        super().__init__()
        self.setWindowTitle("PyQtGraph")
        self.setGeometry(100, 100, 600, 500)
        icon = QIcon("skin.png")
        self.setWindowIcon(icon)
        self.z = z
        if self.z is None:
            self.pos = np.stack([x,y]).T
        else:
            self.pos = np.stack([x,y,z]).T
        self.normalize = normalize
        if self.normalize:
            self.normalize = np.mean(self.pos,axis=0)
            self.pos = self.pos - self.normalize
            
        self.cs = c
        self.s = s
        self.audios = audios
        self.last_point = None

        if self.audios is None:
            self.audios = [None]*len(self.pos)
        if self.cs is None:
            self.cs = [None]*len(self.pos)
        else:
            self.cs = [pg.mkBrush(color=c) for c in self.cs]
        if self.s is None:
            self.s = [7]*len(self.pos)

        self.UiComponents()
        self.show()
 
    def UiComponents(self):
        self.widget = QWidget()
        
        if self.z is None:
            self.plot = pg.plot()
            self.scatter = pg.ScatterPlotItem(
                size=10, brush=pg.mkBrush(255, 255, 255, 120))
            spots = [{'pos': self.pos[i], 
                  'brush': self.cs[i],
                  'size': self.s[i]} for i in range(len(self.pos))]
            self.scatter.addPoints(spots,data=self.audios)
        else:
            self.plot = PickableGLViewWidget(self.pos)
            self.plot.pickedPoint.connect(self.onPoints3DClicked)
            self.plot.show()
            

            self.scatter = GLScatterPlotItem(size=10, color=(255, 255, 255, 120))
            self.scatter.setData(pos = self.pos,
                            color = np.array([[c.color().red()/255.,c.color().green()/255.,c.color().blue()/255.,0.2] for c in self.cs]))
            self.plot.addItem(GLGridItem())
            self.plot.addItem(GLAxisItem())
        self.plot.addItem(self.scatter)
        layout = QGridLayout()
        self.widget.setLayout(layout)
        layout.addWidget(self.plot, 0, 1, 3, 1)
        self.setCentralWidget(self.widget)
        if self.z is None:
            self.scatter.sigClicked.connect(self.onPointsClicked)
        self.text_data = QLabel('Hola',parent=self.widget,)
        self.text_data.setStyleSheet("background-color: rgba(0,0,0,0%); color: rgba(82,255,51)")
        self.text_data.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.text_data.setMaximumSize(self.widget.size().width(),40)
        self.text_data.move(10,self.widget.size().height()-self.text_data.size().height()-10)

    def resizeEvent(self, a0):
        out = super().resizeEvent(a0)
        self.text_data.move(10,self.widget.size().height()-self.text_data.size().height()-10)
        return out

    def onPoints3DClicked(self, idx):
        if idx < len(self.pos):
            if self.audios[idx] is not None:
                self.playAudio(self.audios[idx])
                if not isinstance(self.normalize,bool):
                    unnormalized_position = self.pos[idx] + self.normalize
                else:
                    unnormalized_position = self.pos[idx]
                self.text_data.setText(' Filename: {}\n Position: {}'.format(self.audios[idx],unnormalized_position))
                self.text_data.adjustSize()
            symbolBrushs = np.array([[c.color().red()/255.,c.color().green()/255.,c.color().blue()/255.,0.2] for c in self.cs])
            self.new_point_color = copy.deepcopy(symbolBrushs[idx])
            symbolBrushs[idx] = [255, 0, 0, 255]
            if (self.last_point is not None) and (self.last_point != idx):
                symbolBrushs[self.last_point] = self.last_point_color
                self.last_point = idx
                self.last_point_color = self.new_point_color
            if self.last_point is None:
                self.last_point = idx
                self.last_point_color = self.new_point_color
            self.scatter.setData(pos = self.pos,
                            color = symbolBrushs)

    def onPointsClicked(self,obj,points):
        data_list = obj.data.tolist()
        point_idx = [i for i,p in enumerate(data_list) if points[0] in p]
        data = obj.getData()
        symbolBrushs = self.cs
        self.new_point_color = copy.deepcopy(symbolBrushs[point_idx[0]].color())
        symbolBrushs[point_idx[0]] = pg.mkBrush(color=(255, 0, 0))
        if (self.last_point is not None) and (self.last_point != point_idx[0]):
            symbolBrushs[self.last_point] = self.last_point_color
            self.last_point = point_idx[0]
            self.last_point_color = pg.mkBrush(color=self.new_point_color)
        if self.last_point is None:
            self.last_point = point_idx[0]
            self.last_point_color = pg.mkBrush(color=self.new_point_color)
        
        self.playAudio(points[0].data())
        spots = [{'pos': d, 'size': s, 'brush': b} for d,b,s in zip(np.stack([data[0],data[1]]).T,symbolBrushs,self.s)]
        obj.setData(spots,data=self.audios)

    def playAudio(self,filename):
        x,fs = sf.read(filename)
        sd.play(x, fs)

def extract_mfcc(x):
    x, fs = sf.read(x)
    y = librosa.feature.mfcc(x, sr=fs, n_mfcc=39)
    return np.mean(y,axis=-1)

audios = list(glob.glob('/home/lpepino/nsynth/nsynth-valid/audio/*.wav'))
audios = np.random.choice(audios,size=5000)

commands = np.array([Path(x).stem.split('_')[0] for x in audios])
colormap = cm.tab20
unique_commands = np.unique(commands)
command_color = {x:[c*255 for c in colormap(int((i*colormap.N)/len(unique_commands)))[:3]] for i,x in enumerate(unique_commands)}
cs = [command_color[x] for x in commands]

print('Extracting MFCCs')

N_COMPONENTS = 3
mfccs = [extract_mfcc(x) for x in tqdm(audios)]
mfccs = np.stack(mfccs)
print('Projecting to {}D'.format(N_COMPONENTS))
model = umap.UMAP(n_components=N_COMPONENTS)
embedding = model.fit_transform(mfccs)

App = QApplication(sys.argv)
if N_COMPONENTS == 3:
    window = Window(embedding[:,0],embedding[:,1],embedding[:,2],c=cs,audios=audios)
else:
    window = Window(embedding[:,0],embedding[:,1],c=cs,audios=audios)

sys.exit(App.exec())