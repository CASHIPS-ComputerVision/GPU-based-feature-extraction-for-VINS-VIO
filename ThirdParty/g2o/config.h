#ifndef G2O_CONFIG_H
#define G2O_CONFIG_H


/* #undef G2O_HAVE_OPENGL */
/* #undef G2O_OPENGL_FOUND */
/* #undef G2O_OPENMP */
/* #undef G2O_SHARED_LIBS */
/* #undef G2O_LGPL_SHARED_LIBS */

// available sparse matrix libraries
/* #undef G2O_HAVE_CHOLMOD */
/* #undef G2O_HAVE_CSPARSE */

#define G2O_CXX_COMPILER "GNU /usr/bin/c++"

#ifdef __cplusplus
#include <g2o/core/eigen_types.h>
#endif

#endif
