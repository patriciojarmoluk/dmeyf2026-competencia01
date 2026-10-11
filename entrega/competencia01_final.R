# %%
# ==============================================================================
# DMEyF 2026 - COMPETENCIA 01 - ENTREGA FINAL
# Patricio Jarmoluk
#
# Este script hace todo el proceso, desde el dataset crudo de la competencia
# hasta el archivo que se sube al bot. No necesita ningún archivo intermedio.
#
# Qué hace, en orden:
#   1. Lee competencia_01_crudo.csv (si no está, lo descarga) y calcula clase_ternaria.
#   2. Calidad de datos: ccajas_depositos viene rota en 202105 y se pasa a NA.
#   3. Ranking con cero fijo, mes por mes, de todas las variables originales.
#   4. Feature engineering histórico, calculado sobre los rankings:
#        lags y deltas de orden 1 y 2, y tendencia y promedio de los últimos 3 meses.
#      El modelo nunca ve valores en pesos: solo rankings y columnas derivadas de rankings.
#   5. Optimización de hiperparámetros: entrena en 202103 y 202104, valida en 202106
#      midiendo AUC, sin undersampling y sin las variables de préstamos personales.
#   6. Curva de ganancia en 202106, para decidir cuántos estímulos enviar.
#   7. Modelo final: entrena en 202103-202106 con las variables de préstamos
#      reincorporadas. Ensamble de 30 LightGBM que solo difieren en la semilla;
#      se promedian sus probabilidades y se predice 202108.
#   8. Genera el archivo de la entrega: los numero_de_cliente con mayor probabilidad.
#
# Dónde corre: en la VM de Google Cloud. Usa las carpetas
#   ~/datasets  y  ~/buckets/b1/exp , y las crea si no existen.
# Cuánto tarda: unas 9 horas (1 h la optimización, media hora la curva de junio, 7 h los 30 modelos).
# Si se corta: se vuelve a correr desde el principio. La optimización terminada, la
#   curva de junio y cada modelo ya entrenado quedan guardados y no se repiten.
# Reproducibilidad: todas las semillas salen de la semilla primigenia (123457) y LightGBM
#   corre en modo determinístico, así que dos corridas dan exactamente el mismo archivo.
# ==============================================================================

# %%
# ------------------------------------------------------------------------------
# 0. LIMPIEZA DE MEMORIA Y LIBRERÍAS
# ------------------------------------------------------------------------------
rm(list=ls(all.names=TRUE))
gc(full=TRUE, verbose=FALSE)

# Si falta alguna librería, se instala
paquetes <- c("data.table", "Rcpp", "bbotk", "mlr3", "mlr3mbo", "mlr3learners", "DiceKriging", "lightgbm")
for (paquete in paquetes) {
  if (!requireNamespace(paquete, quietly= TRUE)) install.packages(paquete, repos= "https://cloud.r-project.org")
}

library(data.table)     # manejo del dataset
library(Rcpp)           # compila la función en C que calcula tendencia y promedio
library(bbotk)          # optimización de hiperparámetros
library(mlr3)
library(mlr3mbo)
library(mlr3learners)
library(DiceKriging)
library(lightgbm)       # el modelo


# %%
# ------------------------------------------------------------------------------
# 1. PARÁMETROS (todo lo que define el experimento está en esta celda)
# ------------------------------------------------------------------------------
PARAM <- list()
PARAM$experimento <- "s1027_vm"          # nombre de la carpeta de trabajo dentro de ~/buckets/b1/exp
PARAM$semilla_primigenia <- 123457

# Dataset crudo de la competencia y dirección de donde se baja si no está
PARAM$dataset_crudo <- "~/datasets/competencia_01_crudo.csv"
PARAM$url_crudo <- "https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv"

# Meses
PARAM$bo_train    <- c(202103, 202104)                   # la optimización entrena acá
PARAM$bo_valid    <- c(202106)                           # y valida acá: dos meses después, igual que la entrega
PARAM$train_final <- c(202103, 202104, 202105, 202106)   # el modelo final entrena acá
PARAM$future      <- c(202108)                           # mes a predecir

# Variables que quedan fuera de la optimización y se reincorporan en el modelo final.
PARAM$prestamos <- c("cprestamos_personales", "mprestamos_personales")

# Cantidad de pruebas de la optimización de hiperparámetros
PARAM$hyperparametertuning$iteraciones <- 12

# Semillas del ensamble final: salen todas de la semilla primigenia, con el método de la cátedra
# (se sortean números primos entre 100.000 y 1.000.000). Con la misma primigenia, la lista es siempre la misma.
primos_entre <- function(desde, hasta) {          # criba de Eratóstenes
  es_primo <- rep(TRUE, hasta)
  es_primo[1] <- FALSE
  for (i in 2:floor(sqrt(hasta))) if (es_primo[i]) es_primo[seq(i * i, hasta, by= i)] <- FALSE
  primos <- which(es_primo)
  primos[primos >= desde]
}
set.seed(PARAM$semilla_primigenia, kind= "Mersenne-Twister", normal.kind= "Inversion", sample.kind= "Rejection")
PARAM$semillas <- sample(primos_entre(100000, 1000000), 50)
PARAM$cantidad_semillas <- 30     # modelos del ensamble final: se usan las primeras 30 semillas de la lista
PARAM$junio_semillas <- 5         # modelos que se promedian para la curva de ganancia de 202106
cat("Semillas del ensamble:", PARAM$semillas, "\n")

# Cantidad de estímulos a enviar
PARAM$cortes <- seq(10000, 13500, by= 250)   # se genera un archivo por cada uno, para compararlos en el bot
PARAM$corte_entrega <- 11250                 # el que se entrega
PARAM$archivo_entrega <- "entrega_final_competencia01.csv"

# Ganancia de la campaña, por cliente al que se le envía el estímulo
PARAM$ganancia_acierto <- 1072500   # si el cliente era BAJA+2
PARAM$costo_estimulo   <-   27500   # si no lo era

# Parámetros de LightGBM que quedan fijos
PARAM$lgbm$param_fijos <-  list(
  boosting= "gbdt",
  objective= "binary",
  metric= "auc",
  feature_pre_filter= FALSE,
  first_metric_only= FALSE,
  boost_from_average= TRUE,
  force_row_wise= TRUE,
  deterministic= TRUE,
  verbosity= -100,
  seed= PARAM$semilla_primigenia,
  max_depth= -1L,
  min_gain_to_split= 0,
  lambda_l1= 0.0,
  lambda_l2= 0.0,
  max_bin= 31L,
  bagging_fraction= 1.0,
  pos_bagging_fraction= 1.0,
  neg_bagging_fraction= 1.0,
  is_unbalance= FALSE,
  scale_pos_weight= 1.0,
  drop_rate= 0.1,
  max_drop= 50,
  skip_drop= 0.5,
  extra_trees= FALSE,
  early_stopping= 0,
  min_data_in_leaf= 0,
  feature_fraction= 0.50,
  feature_fraction_bynode= 0.20,
  learning_rate= 0.005
)

# Parámetros de LightGBM que se optimizan, con sus bordes
PARAM$hypeparametertuning$hs <- ps(
  num_iterations= p_int(lower= 600L,  upper= 3500L),
  num_leaves= p_int(lower= 16L,  upper= 128L),
  min_sum_hessian_in_leaf= p_dbl( lower=5e-04, upper=0.03)
)


# %%
# ------------------------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ------------------------------------------------------------------------------

# Ranking con cero fijo de un vector (una variable, en un mes):
#   positivos -> (0, 1]    negativos -> [-1, 0)    cero -> 0    NA -> NA
ranking_cero_fijo <- function(x) {
  x   <- as.numeric(x)
  res <- x                      # así los 0 quedan en 0 y los NA en NA
  pos <- which(x > 0)
  neg <- which(x < 0)
  # Positivos: proporción de positivos del mes con valor <= x  (el mayor = 1)
  if (length(pos) > 0) res[pos] <-  frank( x[pos], ties.method = "average") / length(pos)
  # Negativos: lo mismo sobre el valor absoluto, con signo menos (el más negativo = -1)
  if (length(neg) > 0) res[neg] <- -frank(-x[neg], ties.method = "average") / length(neg)
  res
}

# AUC a partir de las probabilidades y la clase 0/1
calcular_auc <- function(prob, clase01) {
  r  <- frank(prob, ties.method = "average")
  n1 <- sum(clase01 == 1L)
  n0 <- length(clase01) - n1
  (sum(r[clase01 == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# Ganancia acumulada, en millones, enviando el estímulo de mayor a menor probabilidad.
# El elemento k del resultado es la ganancia de enviar a los k clientes con mayor probabilidad.
curva_ganancia <- function(prob, clase_ternaria) {
  gan <- ifelse(clase_ternaria == "BAJA+2", PARAM$ganancia_acierto, -PARAM$costo_estimulo)
  cumsum(gan[order(-prob)]) / 1e6
}


# %%
# ------------------------------------------------------------------------------
# 3. CARPETAS Y LECTURA DEL DATASET CRUDO
# ------------------------------------------------------------------------------
dir.create("~/datasets", showWarnings=FALSE, recursive=TRUE)
dir.create("~/buckets/b1/exp", showWarnings=FALSE, recursive=TRUE)

if (!file.exists(path.expand(PARAM$dataset_crudo))) {
  cat("Descargando el dataset crudo...\n")
  options(timeout= 3600)
  download.file(PARAM$url_crudo, destfile= path.expand(PARAM$dataset_crudo), mode= "wb")
}

setwd("~/buckets/b1/exp")
dir.create(PARAM$experimento, showWarnings=FALSE)
setwd(paste0("~/buckets/b1/exp/", PARAM$experimento))

cat("Leyendo dataset crudo...\n")
dataset <- fread(PARAM$dataset_crudo, integer64 = "double")
cat("Filas:", nrow(dataset), "| columnas:", ncol(dataset), "\n")
stopifnot(all(c(PARAM$bo_train, PARAM$bo_valid, PARAM$train_final, PARAM$future, 202107) %in% unique(dataset$foto_mes)))


# %%
# ------------------------------------------------------------------------------
# 3a. CLASE TERNARIA
#   BAJA+1:   el cliente no está el mes siguiente
#   BAJA+2:   está el mes siguiente pero no a los dos meses
#   CONTINUA: sigue siendo cliente
# ------------------------------------------------------------------------------
if ("clase_ternaria" %in% names(dataset)) dataset[, clase_ternaria := NULL]

# calculo el periodo0 consecutivo
dsimple <- dataset[, list(
    "pos" = .I,
    numero_de_cliente,
    periodo0 = as.integer(foto_mes/100)*12 +  foto_mes%%100 ) ]

# ordeno
setorder( dsimple, numero_de_cliente, periodo0 )

# calculo topes
periodo_ultimo <- dsimple[, max(periodo0) ]
periodo_anteultimo <- periodo_ultimo - 1

# calculo los leads de orden 1 y 2
dsimple[, c("periodo1", "periodo2") :=
    shift(periodo0, n=1:2, fill=NA, type="lead"),  numero_de_cliente ]

# assign most common class values = "CONTINUA"
dsimple[ periodo0 < periodo_anteultimo, clase_ternaria := "CONTINUA" ]

# calculo BAJA+1
dsimple[ periodo0 < periodo_ultimo &
    ( is.na(periodo1) | periodo0 + 1 < periodo1 ),
    clase_ternaria := "BAJA+1" ]

# calculo BAJA+2
dsimple[ periodo0 < periodo_anteultimo & (periodo0+1 == periodo1 )
    & ( is.na(periodo2) | periodo0 + 2 < periodo2 ),
    clase_ternaria := "BAJA+2" ]

# pego el resultado en el dataset original
setorder( dsimple, pos )
dataset[, clase_ternaria := dsimple$clase_ternaria ]
rm(dsimple)

print(dataset[, .N, keyby = list(foto_mes, clase_ternaria)])
stopifnot(dataset[foto_mes %in% PARAM$train_final & is.na(clase_ternaria), .N] == 0)


# %%
# ------------------------------------------------------------------------------
# 3b. CALIDAD DE DATOS: ccajas_depositos viene rota en 202105 -> NA
# Se hace antes del ranking, para que el ranking no tome ese valor roto como real.
# ------------------------------------------------------------------------------
dataset[foto_mes == 202105, ccajas_depositos := NA]


# %%
# ------------------------------------------------------------------------------
# 3c. RANKING CON CERO FIJO DE TODAS LAS VARIABLES ORIGINALES, MES POR MES
# Cada variable se reemplaza por su ranking dentro del mes (conserva el nombre).
# Así un mismo valor significa lo mismo en todos los meses, aunque haya inflación
# o meses atípicos como junio, que tiene el aguinaldo.
# ------------------------------------------------------------------------------
cols_base <- setdiff(names(dataset), c("numero_de_cliente", "foto_mes", "clase_ternaria"))
cat("Variables originales a rankear:", length(cols_base), "(incluye las 2 de préstamos)\n")
stopifnot(all(PARAM$prestamos %in% cols_base), all(sapply(cols_base, function(v) is.numeric(dataset[[v]]))))

# Control: la cantidad de ceros y de NA tiene que ser la misma antes y después del ranking
control_antes <- dataset[, .(ceros = sum(sapply(.SD, function(v) sum(v == 0, na.rm = TRUE))),
                             nas   = sum(sapply(.SD, function(v) sum(is.na(v))))),
                         .SDcols = cols_base]

dataset[, (cols_base) := lapply(.SD, as.numeric), .SDcols = cols_base]
dataset[, (cols_base) := lapply(.SD, ranking_cero_fijo), by = foto_mes, .SDcols = cols_base]

control_despues <- dataset[, .(ceros = sum(sapply(.SD, function(v) sum(v == 0, na.rm = TRUE))),
                               nas   = sum(sapply(.SD, function(v) sum(is.na(v))))),
                           .SDcols = cols_base]
print(rbind(antes = control_antes, despues = control_despues))
stopifnot(identical(control_antes, control_despues))
stopifnot(all(sapply(cols_base, function(v) all(abs(dataset[[v]]) <= 1, na.rm = TRUE))))


# %%
# ------------------------------------------------------------------------------
# 3d. LAGS Y DELTAS DE ORDEN 1 Y 2, CALCULADOS SOBRE LOS RANKINGS
#   lag_k   = ranking del mismo cliente exactamente k meses antes (NA si ese mes no está)
#   delta_k = ranking actual - lag_k
# ------------------------------------------------------------------------------
periodo <- dataset[, as.integer(foto_mes/100)*12 + foto_mes%%100]
clave   <- as.numeric(dataset$numero_de_cliente) * 100 + (periodo - min(periodo))   # identifica cliente y mes
stopifnot(anyDuplicated(clave) == 0, max(periodo) - min(periodo) < 98)
fila_lag1 <- match(clave - 1, clave)   # fila del mismo cliente el mes anterior
fila_lag2 <- match(clave - 2, clave)   # fila del mismo cliente dos meses antes
rm(periodo, clave)

for (campo in cols_base) {
  r <- dataset[[campo]]
  set(dataset, j = paste0(campo, "_lag1"),   value = r[fila_lag1])
  set(dataset, j = paste0(campo, "_delta1"), value = r - r[fila_lag1])
  set(dataset, j = paste0(campo, "_lag2"),   value = r[fila_lag2])
  set(dataset, j = paste0(campo, "_delta2"), value = r - r[fila_lag2])
}
rm(fila_lag1, fila_lag2, r)
gc(full=TRUE, verbose=FALSE)


# %%
# ------------------------------------------------------------------------------
# 3e. TENDENCIA Y PROMEDIO DE LOS ÚLTIMOS 3 MESES, CALCULADOS SOBRE LOS RANKINGS
# Las dos funciones que siguen son el código de Feature Engineering Histórico
# que publicó Gustavo en Zulip, sin modificar.
# Para cada variable original se agregan:
#   <variable>_tend3 : pendiente de la recta que ajusta los últimos 3 meses
#   <variable>_avg3  : promedio de los últimos 3 meses
# ------------------------------------------------------------------------------


cppFunction("NumericVector fhistC(NumericVector pcolumna, IntegerVector pdesde )
{
  /* Aqui se cargan los valores para la regresion */
  double  x[100] ;
  double  y[100] ;

  int n = pcolumna.size();
  NumericVector out( 5*n );

  for(int i = 0; i < n; i++)
  {
    //lag
    if( pdesde[i]-1 < i )  out[ i + 4*n ]  =  pcolumna[i-1] ;
    else                   out[ i + 4*n ]  =  NA_REAL ;


    int  libre    = 0 ;
    int  xvalor   = 1 ;

    for( int j= pdesde[i]-1;  j<=i; j++ )
    {
       double a = pcolumna[j] ;

       if( !R_IsNA( a ) )
       {
          y[ libre ]= a ;
          x[ libre ]= xvalor ;
          libre++ ;
       }

       xvalor++ ;
    }

    /* Si hay al menos dos valores */
    if( libre > 1 )
    {
      double  xsum  = x[0] ;
      double  ysum  = y[0] ;
      double  xysum = xsum * ysum ;
      double  xxsum = xsum * xsum ;
      double  vmin  = y[0] ;
      double  vmax  = y[0] ;

      for( int h=1; h<libre; h++)
      {
        xsum  += x[h] ;
        ysum  += y[h] ;
        xysum += x[h]*y[h] ;
        xxsum += x[h]*x[h] ;

        if( y[h] < vmin )  vmin = y[h] ;
        if( y[h] > vmax )  vmax = y[h] ;
      }

      out[ i ]  =  (libre*xysum - xsum*ysum)/(libre*xxsum -xsum*xsum) ;
      out[ i + n ]    =  vmin ;
      out[ i + 2*n ]  =  vmax ;
      out[ i + 3*n ]  =  ysum / libre ;
    }
    else
    {
      out[ i       ]  =  NA_REAL ;
      out[ i + n   ]  =  NA_REAL ;
      out[ i + 2*n ]  =  NA_REAL ;
      out[ i + 3*n ]  =  NA_REAL ;
    }
  }

  return  out;
}")
#------------------------------------------------------------------------------


TendenciaYmuchomas <- function(
    dataset, cols, ventana = 6, tendencia = TRUE,
    minimo = TRUE, maximo = TRUE, promedio = TRUE,
    ratioavg = FALSE, ratiomax = FALSE) {
  gc(verbose= FALSE)
  # Esta es la cantidad de meses que utilizo para la historia
  ventana_regresion <- ventana

  last <- nrow(dataset)

  # creo el vector_desde que indica cada ventana
  # de esta forma se acelera el procesamiento ya que lo hago una sola vez
  vector_ids <- dataset[ , numero_de_cliente ]

  vector_desde <- seq(
    -ventana_regresion + 2,
    nrow(dataset) - ventana_regresion + 1
  )

  vector_desde[1:ventana_regresion] <- 1

  for (i in 2:last) {
    if (vector_ids[i - 1] != vector_ids[i]) {
      vector_desde[i] <- i
    }
  }
  for (i in 2:last) {
    if (vector_desde[i] < vector_desde[i - 1]) {
      vector_desde[i] <- vector_desde[i - 1]
    }
  }

  for (campo in cols) {
    nueva_col <- fhistC(dataset[, get(campo)], vector_desde)

    if (tendencia) {
      dataset[, paste0(campo, "_tend", ventana) :=
        nueva_col[(0 * last + 1):(1 * last)]]
    }

    if (minimo) {
      dataset[, paste0(campo, "_min", ventana) :=
        nueva_col[(1 * last + 1):(2 * last)]]
    }

    if (maximo) {
      dataset[, paste0(campo, "_max", ventana) :=
        nueva_col[(2 * last + 1):(3 * last)]]
    }

    if (promedio) {
      dataset[, paste0(campo, "_avg", ventana) :=
        nueva_col[(3 * last + 1):(4 * last)]]
    }

    if (ratioavg) {
      dataset[, paste0(campo, "_ratioavg", ventana) :=
        get(campo) / nueva_col[(3 * last + 1):(4 * last)]]
    }

    if (ratiomax) {
      dataset[, paste0(campo, "_ratiomax", ventana) :=
        get(campo) / nueva_col[(2 * last + 1):(3 * last)]]
    }
  }
}
#------------------------------------------------------------------------------

columnas_antes <- ncol(dataset)
# El cálculo necesita el dataset ordenado por cliente y mes.
# Guardo el orden original de las filas y lo restauro al terminar.
dataset[, pos_original := .I]
setorderv(dataset, c("numero_de_cliente", "foto_mes"))

TendenciaYmuchomas(dataset,
  cols= cols_base,
  ventana= 3,
  tendencia= TRUE,
  minimo= FALSE,
  maximo= FALSE,
  promedio= TRUE,
  ratioavg= FALSE,
  ratiomax= FALSE
)

setorder(dataset, pos_original)
dataset[, pos_original := NULL]
cat("Columnas nuevas de tendencia y promedio:", ncol(dataset) - columnas_antes, "\n")
stopifnot(ncol(dataset) - columnas_antes == 2 * length(cols_base))
gc(full=TRUE, verbose=FALSE)


# %%
# ------------------------------------------------------------------------------
# 4. PREPARACIÓN DE DATOS
# ------------------------------------------------------------------------------
# Clase binaria que aprende el modelo: 1 = BAJA+1 o BAJA+2, 0 = CONTINUA
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+1", "BAJA+2"), 1L, 0L)]

# Variables predictoras: por cada variable original, su ranking, lag1, delta1, lag2, delta2, tend3 y avg3.
# numero_de_cliente y foto_mes NO son predictoras.
campos_todos     <- setdiff(colnames(dataset), c("numero_de_cliente", "foto_mes", "clase_ternaria", "clase01"))
campos_prestamos <- grep(paste0("^(", paste(PARAM$prestamos, collapse = "|"), ")(_lag[12]|_delta[12]|_tend3|_avg3)?$"), campos_todos, value = TRUE)
campos_bo        <- setdiff(campos_todos, campos_prestamos)    # las de la optimización: todas menos las de préstamos
cat("Predictoras totales:", length(campos_todos), "| de préstamos:", length(campos_prestamos),
    "| para la optimización:", length(campos_bo), "\n")
stopifnot(length(campos_todos) == 7 * length(cols_base), length(campos_prestamos) == 14)


# %%
# ------------------------------------------------------------------------------
# 5. OPTIMIZACIÓN DE HIPERPARÁMETROS
# Entrena en 202103 y 202104 con todos los clientes (sin undersampling) y sin las
# variables de préstamos. Valida en 202106 completo, midiendo AUC.
# Son 12 pruebas, que es el diseño inicial de mlr3mbo para 3 hiperparámetros:
# 12 combinaciones repartidas al azar dentro de los bordes. Se elige la de mayor AUC.
# ------------------------------------------------------------------------------
dtrain <- lgb.Dataset(
  data= data.matrix(dataset[foto_mes %in% PARAM$bo_train, campos_bo, with= FALSE]),
  label= dataset[foto_mes %in% PARAM$bo_train, clase01],
  free_raw_data= FALSE
)
mvalidate      <- data.matrix(dataset[foto_mes %in% PARAM$bo_valid, campos_bo, with= FALSE])
validate_y     <- dataset[foto_mes %in% PARAM$bo_valid, clase01]
validate_clase <- dataset[foto_mes %in% PARAM$bo_valid, clase_ternaria]

# Si la optimización ya está terminada en la carpeta de trabajo, no se repite
bo_hecha <- file.exists("BO_log.txt") && nrow(fread("BO_log.txt")) >= PARAM$hyperparametertuning$iteraciones
if (!bo_hecha) {
  if (file.exists("BO_log.txt")) file.remove("BO_log.txt")    # quedó a medias: se empieza de nuevo
  giter <- 0

  # Entrena un modelo con los hiperparámetros xs y devuelve su AUC en 202106
  Estimar_lightgbm <- function(xs) {
    flush.console()
    giter <<- giter + 1
    t0 <- Sys.time()

    param_completo <- modifyList(PARAM$lgbm$param_fijos, xs)
    modelo <- lgb.train(data= dtrain, param= param_completo)
    prob <- predict(modelo, mvalidate)
    ganancia <- curva_ganancia(prob, validate_clase)              # solo informativo, no se usa para elegir
    registro <- data.table(
      iter= giter, num_iterations= xs$num_iterations, num_leaves= xs$num_leaves,
      min_sum_hessian_in_leaf= xs$min_sum_hessian_in_leaf,
      metrica= calcular_auc(prob, validate_y),                    # lo que se optimiza
      ganancia_10500= ganancia[10500], ganancia_11000= ganancia[11000],
      ganancia_max= max(ganancia), envios_max= which.max(ganancia),
      minutos= round(as.numeric(difftime(Sys.time(), t0, units= "mins")), 1))
    fwrite(registro, file= "BO_log.txt", sep= "\t", append= file.exists("BO_log.txt"))
    print(registro)

    set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
    return( list( metrica = registro$metrica) )
  }

  cat("Iniciando Optimización Bayesiana...\n")
  objective <- ObjectiveRFun$new(
    fun= Estimar_lightgbm,
    domain= PARAM$hypeparametertuning$hs,
    codomain= ps(metrica = p_dbl(tags="maximize")),
    id= "BO_lightgbm"
  )

  instance <- OptimInstanceBatchSingleCrit$new(
    objective= objective,
    terminator=  trm("evals", n_evals = PARAM$hyperparametertuning$iteraciones)
  )

  surrogate <- srlrn(lrn("regr.km",
                         optim.method = "BFGS",
                         nugget.estim = TRUE,
                         control= list(trace = FALSE)))

  acq_function  <- acqf("ei")
  acq_optimizer  <- acqo(
    optimizer  = opt("random_search", batch_size= 1000),
    terminator = trm("evals", n_evals= 1000)
  )

  optimizer <- opt("mbo",
                   loop_function = bayesopt_ego,
                   surrogate= surrogate,
                   acq_function= acq_function,
                   acq_optimizer= acq_optimizer
  )

  set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
  bayesiana_salida <- optimizer$optimize(instance)
}

# Los hiperparámetros del modelo final son los de la prueba con mayor AUC
tb_bayesiana <- fread("BO_log.txt")
setorder(tb_bayesiana, -metrica)
cat("\nPruebas de la optimización, de mayor a menor AUC en 202106:\n")
print(tb_bayesiana)
PARAM$out$lgbm$mejores_hiperparametros <- as.list(
  tb_bayesiana[1, list(num_iterations, num_leaves, min_sum_hessian_in_leaf)])

param_final <- modifyList(PARAM$lgbm$param_fijos, PARAM$out$lgbm$mejores_hiperparametros)
cat("\nHiperparámetros del modelo final -> num_iterations:", param_final$num_iterations,
    "| num_leaves:", param_final$num_leaves,
    "| min_sum_hessian_in_leaf:", format(param_final$min_sum_hessian_in_leaf, digits = 15), "\n")

semillas_ensamble <- PARAM$semillas[1:PARAM$cantidad_semillas]
stopifnot(length(semillas_ensamble) == PARAM$cantidad_semillas, anyDuplicated(semillas_ensamble) == 0)


# %%
# ------------------------------------------------------------------------------
# 6. CURVA DE GANANCIA EN 202106
# Con los hiperparámetros elegidos se entrenan 5 modelos en 202103 y 202104 (sin
# préstamos, igual que en la optimización), se promedian y se predice 202106, que
# tiene la clase conocida. Muestra cuánto se ganaría según cuántos estímulos se envíen.
# Es solo para mirar el punto de corte: no interviene en el archivo de la entrega.
# ------------------------------------------------------------------------------
if (PARAM$junio_semillas > 0 && !file.exists("curva_junio.txt")) {
  prob_junio <- rep(0, nrow(mvalidate))
  for (i in 1:PARAM$junio_semillas) {
    cat(">> Junio: modelo", i, "de", PARAM$junio_semillas, "|", format(Sys.time(), "%H:%M"), "\n")
    param_final$seed <- PARAM$semillas[i]      # las primeras semillas de la lista
    prob_junio <- prob_junio + predict(lgb.train(data = dtrain, param = param_final), mvalidate)
  }
  ganancia_junio <- curva_ganancia(prob_junio, validate_clase)
  tb_junio <- data.table(envios = seq(8000, 15000, by = 500))
  tb_junio[, ganancia_millones := round(ganancia_junio[envios], 1)]
  cat("\n=== GANANCIA EN 202106 (entrenando en 202103 y 202104, ensamble de", PARAM$junio_semillas, "semillas) ===\n")
  print(tb_junio)
  cat("Máximo:", round(max(ganancia_junio), 1), "millones en", which.max(ganancia_junio), "envíos\n")
  fwrite(tb_junio, file = "curva_junio.txt", sep = "\t")
}
rm(dtrain, mvalidate)
gc(full=TRUE, verbose=FALSE)


# %%
# ------------------------------------------------------------------------------
# 7. MODELO FINAL: ENSAMBLE DE SEMILLAS
# Entrena en 202103-202106 con todos los clientes y con las variables de préstamos
# reincorporadas. Todos los modelos usan los mismos hiperparámetros: lo único que
# cambia entre uno y otro es la semilla. Cada modelo predice 202108 y guarda sus
# probabilidades en modelos/prob_<semilla>.txt
# ------------------------------------------------------------------------------
dtrain_final <- lgb.Dataset(
  data = data.matrix(dataset[foto_mes %in% PARAM$train_final, campos_todos, with = FALSE]),
  label = dataset[foto_mes %in% PARAM$train_final, clase01]
)
filas_train_final <- dataset[foto_mes %in% PARAM$train_final, .N]
mfuture <- data.matrix(dataset[foto_mes %in% PARAM$future, campos_todos, with = FALSE])
clientes_future <- dataset[foto_mes %in% PARAM$future, numero_de_cliente]
stopifnot(anyDuplicated(clientes_future) == 0)
rm(dataset)
gc(full=TRUE, verbose=FALSE)

probabilidades_acumuladas <- rep(0, length(clientes_future))

cat("\n====================================================================\n")
cat("INICIANDO ENSAMBLE - CANTIDAD DE MODELOS:", length(semillas_ensamble),
    "| filas:", filas_train_final, "| predictoras:", length(campos_todos), "\n")
cat("====================================================================\n")

dir.create("modelos", showWarnings=FALSE)
for (i in seq_along(semillas_ensamble)) {
  semilla_actual <- semillas_ensamble[i]
  archivo_prob <- paste0("modelos/prob_", semilla_actual, ".txt")

  if (file.exists(archivo_prob)) {
    cat(">> Modelo", i, "de", length(semillas_ensamble), "| Semilla:", semilla_actual, "ya estaba hecho\n")
  } else {
    cat(">> Entrenando modelo", i, "de", length(semillas_ensamble), "| Semilla:", semilla_actual,
        "|", format(Sys.time(), "%H:%M"), "...\n")
    param_final$seed <- semilla_actual
    modelo_iter <- lgb.train(
      data = dtrain_final,
      param = param_final
    )
    prediccion_iter <- predict(modelo_iter, mfuture)
    fwrite(data.table(numero_de_cliente = clientes_future, prob = prediccion_iter), file = archivo_prob, sep = "\t")
  }

  # Las probabilidades se leen siempre del archivo y se unen por numero_de_cliente.
  # Así el resultado es idéntico si el modelo se entrenó recién o en una corrida anterior.
  tb_modelo <- fread(archivo_prob)
  prediccion_iter <- tb_modelo$prob[match(clientes_future, tb_modelo$numero_de_cliente)]
  stopifnot(nrow(tb_modelo) == length(clientes_future), !anyNA(prediccion_iter))
  probabilidades_acumuladas <- probabilidades_acumuladas + prediccion_iter
}

# Promedio de las probabilidades de todos los modelos
probabilidad_promedio <- probabilidades_acumuladas / length(semillas_ensamble)

tb_prediccion <- data.table(numero_de_cliente = clientes_future, prob = probabilidad_promedio)
setorder(tb_prediccion, -prob)     # de mayor a menor probabilidad de baja


# %%
# ------------------------------------------------------------------------------
# 8. ARCHIVOS DE SALIDA
# Formato que pide el bot: una sola columna con numero_de_cliente, sin encabezado,
# números enteros, entre 8.000 y 15.000 filas.
# ------------------------------------------------------------------------------
options(scipen = 999)
fwrite(tb_prediccion, file = paste0(PARAM$experimento, "_probabilidades.txt"), sep = "\t")

# Un archivo por cada cantidad de envíos, para comparar los puntos de corte en el bot
for (corte in PARAM$cortes) {
  tb_entrega <- tb_prediccion[1:corte, list(numero_de_cliente = as.integer(numero_de_cliente))]
  nombre_submit <- paste0(PARAM$experimento, "_", corte, "envios.csv")
  fwrite(tb_entrega, file = nombre_submit, sep = ",", col.names = FALSE, quote = FALSE)
  cat("Archivo generado:", nombre_submit, "\n")
}

# EL ARCHIVO DE LA ENTREGA
tb_entrega <- tb_prediccion[1:PARAM$corte_entrega, list(numero_de_cliente = as.integer(numero_de_cliente))]
fwrite(tb_entrega, file = PARAM$archivo_entrega, sep = ",", col.names = FALSE, quote = FALSE)

# Control de formato del archivo de la entrega
lineas <- readLines(PARAM$archivo_entrega)
stopifnot(length(lineas) == PARAM$corte_entrega, length(lineas) >= 8000, length(lineas) <= 15000,
          all(grepl("^[0-9]+$", lineas)), anyDuplicated(lineas) == 0,
          all(as.integer(lineas) %in% clientes_future))

cat("\n====================================================================\n")
cat("ENTREGA GENERADA:", file.path(getwd(), PARAM$archivo_entrega), "\n")
cat("Modelos en el ensamble:", length(semillas_ensamble), "| estímulos a enviar:", PARAM$corte_entrega, "\n")
cat("====================================================================\n")

# Versiones con las que se corrió
cat("\n", R.version.string, "| lightgbm", as.character(packageVersion("lightgbm")),
    "| data.table", as.character(packageVersion("data.table")),
    "| mlr3mbo", as.character(packageVersion("mlr3mbo")), "\n")
