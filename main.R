# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
library(tidyverse)
library(skimr)
library(recipes)
library(h2o)
application_train_tbl <- read_csv('../input/application_train.csv')
application_test_tbl  <- read_csv('../input/application_test.csv')
    
# Training data: Separate into x and y tibbles
x_train_tbl <- select(application_train_tbl, -TARGET)
y_train_tbl <-  select(application_train_tbl, TARGET)  


# Testing data: What we submit in the competition
x_test_tbl  <- application_test_tbl

# Remove the original data to save memory
rm(application_train_tbl)
rm(application_test_tbl)

skim_to_list(x_train_tbl)

string_2_factor_names <- x_train_tbl %>%
    select_if(is.character) %>%
    names()

string_2_factor_names


unique_numeric_values_tbl <- x_train_tbl %>%
    select_if(is.numeric) %>%
    map_df(~ unique(.) %>% length()) %>%
    gather() %>%
    arrange(value) %>%
    mutate(key = as_factor(key))

unique_numeric_values_tbl

factor_limit <- 7

num_2_factor_names <- unique_numeric_values_tbl %>%
    filter(value < factor_limit) %>%
    arrange(desc(value)) %>%
    pull(key) %>%
    as.character()

num_2_factor_names

missing_tbl <- x_train_tbl %>%
    summarize_all(.funs = ~ sum(is.na(.)) / length(.)) %>%
    gather() %>%
    arrange(desc(value)) %>%
    filter(value > 0)

missing_tbl

rec_obj <- recipe(~ ., data = x_train_tbl) %>%
    step_string2factor(string_2_factor_names) %>%
    step_num2factor(num_2_factor_names) %>%
    step_meanimpute(all_numeric()) %>%
    step_modeimpute(all_nominal()) %>%
    prep(stringsAsFactors = FALSE)

rec_obj

x_train_processed_tbl <- bake(rec_obj, x_train_tbl) 
x_test_processed_tbl  <- bake(rec_obj, x_test_tbl)

# Before transformation
x_train_tbl %>%
    select(1:30) %>%
    glimpse()
    
    # After transformation
x_train_processed_tbl %>%
    select(1:30) %>%
    glimpse()
    
    
y_train_processed_tbl <- y_train_tbl %>%
    mutate(TARGET = TARGET %>% as.character() %>% as.factor())
    
rm(rec_obj)
rm(x_train_tbl)
rm(x_test_tbl)
rm(y_train_tbl)

h2o.init()

h2o.no_progress()

data_h2o <- as.h2o(bind_cols(y_train_processed_tbl, x_train_processed_tbl))

splits_h2o <- h2o.splitFrame(data_h2o, ratios = c(0.7, 0.15), seed = 1234)

train_h2o <- splits_h2o[[1]]
valid_h2o <- splits_h2o[[2]]
test_h2o  <- splits_h2o[[3]]

y <- "TARGET"
x <- setdiff(names(train_h2o), y)

automl_models_h2o <- h2o.automl(
    x = x,
    y = y,
    training_frame    = train_h2o,
    validation_frame  = valid_h2o,
    leaderboard_frame = test_h2o,
    max_runtime_secs  = 90
)

automl_leader <- automl_models_h2o@leader

performance_h2o <- h2o.performance(automl_leader, newdata = test_h2o)

performance_h2o %>%
    h2o.confusionMatrix()
    
performance_h2o %>%
    h2o.auc()
    
prediction_h2o <- h2o.predict(automl_leader, newdata = as.h2o(x_test_processed_tbl)) 

prediction_tbl <- prediction_h2o %>%
    as.tibble() %>%
    bind_cols(
        x_test_processed_tbl %>% select(SK_ID_CURR)
    ) %>%
    select(SK_ID_CURR, p1) %>%
    rename(TARGET = p1)

prediction_tbl

prediction_tbl %>%
    write_csv(path = "submission_001.csv")