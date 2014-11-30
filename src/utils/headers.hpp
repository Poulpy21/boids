#ifndef _CUSTOM_HEADERS_H
#define _CUSTOM_HEADERS_H

#include <mpi.h>

#include <GL/glew.h> //1st gl include (mandatory)
#include <GL/glut.h>

// QT
#include <QApplication>
#include <QtGui>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QElapsedTimer>
#include <QMainWindow>
#include <QObject>
#include <QWidget>
#include <QKeyEvent>
#include <QMessageBox>
#include <QDialog>
#include <QFileDialog>
#include <QMenuBar>
#include <QStatusBar>
#include <QProgressBar>
#include <QLabel>
#include <QBoxLayout>
#include <QHBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QListWidget>
#include <QListWidgetItem>
#include <QGraphicsView>
#include <QGLWidget>
#include <QGLFormat>
#include <QGLContext>
#include <QGraphicsScene>
#include <QPainter>
#include <QPaintEngine>
#include <QRect>
#include <QRectF>
#include <QResizeEvent>
#include <QTimer>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QGenericMatrix>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <QGLViewer/qglviewer.h>
#include <QGLViewer/camera.h>
#include <QGLViewer/quaternion.h>

#include "defines.hpp"
#include "log.hpp"

#include "utils.hpp"
#include "glUtils.hpp"
#include "cudaUtils.hpp"

#include "vec.hpp"
#include "matrix.hpp"
#include "types.hpp"

#include "consts.hpp"
#include "globals.hpp"

#endif /* end of include guard: HEADERS_H */

