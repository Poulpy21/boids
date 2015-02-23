#ifndef _CUSTOM_HEADERS_H
#define _CUSTOM_HEADERS_H

#include "defines.hpp"

// Start NVCC proof
#ifndef __CUDACC__

#ifdef GUI_ENABLED
#include <GL/glew.h> //1st gl include (mandatory)
#include <GL/glut.h>
#endif

#include <mpi.h>

// QT
#ifdef GUI_ENABLED
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

#include <QGLViewer/qglviewer.h>
#include <QGLViewer/camera.h>
#include <QGLViewer/quaternion.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>
#endif

#endif
// end NVCC proof

#ifdef CUDA_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

#ifdef CURAND_ENABLED
#include <curand.h>
#endif

#ifdef THRUST_ENABLED
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//#ifdef __CUDACC__
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
//#endif
#endif

// Start NVCC proof
#ifndef __CUDACC__

#ifdef GUI_ENABLED
#include "glUtils.hpp"
#endif

#endif
// end NVCC proof

#include "utils.hpp"
#include "log.hpp"

#ifdef CUDA_ENABLED
#include "cudaUtils.hpp"
#endif

#include "types.hpp"

#include "vec.hpp"
#include "vec3.hpp"
#include "globals.hpp"

// Start NVCC proof (C++11 everywhere !)
//
#ifndef __CUDACC__
#include "vec2.hpp"
#include "vecBool.hpp"
#include "matrix.hpp"
#include "consts.hpp"
#endif

// end NVCC proof

#include <exception> //for NOT_IMPLEMENTED_YET macro
#include <typeinfo>

#endif /* end of include guard: HEADERS_H */

