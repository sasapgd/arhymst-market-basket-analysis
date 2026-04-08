# ============================================================
# MARKET BASKET ANALYSIS (R) -> Rules for Python
# ============================================================

# ------------------------------------------------------------
# PATH RESOLUTION
# ------------------------------------------------------------

get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  matched <- grep(file_arg, cmd_args, value = TRUE)

  if (length(matched) > 0) {
    return(dirname(normalizePath(sub(file_arg, "", matched[1]), winslash = "/")))
  }

  frame_file <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) NULL)
  if (!is.null(frame_file)) {
    return(dirname(normalizePath(frame_file, winslash = "/")))
  }

  normalizePath(getwd(), winslash = "/")
}

library(data.table)
library(arules)

setDTthreads(0)

# ------------------------------------------------------------
# PATH SETTINGS
# ------------------------------------------------------------

script_dir <- get_script_dir()
input_folder <- file.path(script_dir, "Data")
output_file <- file.path(script_dir, "Rules_For_Python.csv")

all_products_file <- file.path(script_dir, "ALL_PRODUCTS.csv")
item_frequency_file <- file.path(script_dir, "ITEM_FREQUENCY.csv")
products_in_rules_file <- file.path(script_dir, "PRODUCTS_IN_RULES.csv")
excluded_rows_file <- file.path(script_dir, "EXCLUDED_ROWS_SUMMARY.csv")

# ------------------------------------------------------------
# APRIORI PARAMETERS
# ------------------------------------------------------------

MIN_SUPPORT <- 0.001
MIN_CONF <- 0.30
MAXLEN_RULE <- 3
MIN_LIFT_KEEP <- 1.0
REMOVE_REDUNDANT <- TRUE

# ------------------------------------------------------------
# COLUMN NAMES
# ------------------------------------------------------------

COL_PERSON <- c("PERSON_PUBLIC_KEY", "PERSON PUBLIC KEY", "Person")
COL_DATE <- c("DATE", "Date")
COL_CHANNEL <- c("CHANNEL", "ONLINE/OFFLINE", "Channel")
COL_PRODUCT <- c("PRODUCT_CATEGORY", "PRODUCT CATEGORY", "Product")

# ------------------------------------------------------------
# LOAD CSV FILES
# ------------------------------------------------------------

files <- list.files(
  path = input_folder,
  pattern = "\\.csv$",
  full.names = TRUE,
  ignore.case = TRUE
)

if (length(files) == 0) {
  stop("No CSV files found.")
}

# ------------------------------------------------------------
# FILE LOADING HELPER
# ------------------------------------------------------------

load_file <- function(file) {
  normalize_name <- function(x) toupper(gsub("[^A-Z0-9]+", "", x))

  resolve_col <- function(columns, candidates, label) {
    idx <- match(normalize_name(candidates), normalize_name(columns))
    idx <- idx[!is.na(idx)]

    if (!length(idx)) {
      stop(
        sprintf(
          "File '%s' does not contain a column for %s. Available columns: %s",
          basename(file),
          label,
          paste(columns, collapse = ", ")
        )
      )
    }

    columns[idx[1]]
  }

  header <- names(fread(file, nrows = 0, showProgress = FALSE))

  person_col <- resolve_col(header, COL_PERSON, "person")
  date_col <- resolve_col(header, COL_DATE, "date")
  channel_col <- resolve_col(header, COL_CHANNEL, "channel")
  product_col <- resolve_col(header, COL_PRODUCT, "product")

  df <- fread(
    file,
    fill = TRUE,
    showProgress = FALSE,
    select = c(person_col, date_col, channel_col, product_col)
  )

  setnames(
    df,
    c(person_col, date_col, channel_col, product_col),
    c("Person", "Date", "Channel", "Product")
  )
  df
}

# ------------------------------------------------------------
# LOAD AND CLEAN INPUT DATA
# ------------------------------------------------------------

dt <- rbindlist(lapply(files, load_file), use.names = TRUE)

dt[, Person := trimws(as.character(Person))]
dt[, Date := trimws(as.character(Date))]
dt[, Channel := trimws(as.character(Channel))]
dt[, Product := trimws(as.character(Product))]

dt[, Channel := toupper(Channel)]
dt[is.na(Channel) | Channel == "", Channel := "UNKNOWN"]

excluded_missing_person <- is.na(dt$Person) | dt$Person == ""
excluded_missing_date <- is.na(dt$Date) | dt$Date == ""
excluded_missing_product <- is.na(dt$Product) | dt$Product == ""
excluded_any <- excluded_missing_person | excluded_missing_date | excluded_missing_product

excluded_summary <- data.table(
  Metric = c(
    "Rows before cleaning",
    "Excluded: missing Person",
    "Excluded: missing Date",
    "Excluded: missing Product",
    "Excluded: total rows removed",
    "Rows kept for baskets"
  ),
  Count = c(
    nrow(dt),
    sum(excluded_missing_person),
    sum(excluded_missing_date),
    sum(excluded_missing_product),
    sum(excluded_any),
    sum(!excluded_any)
  )
)

fwrite(excluded_summary, excluded_rows_file, sep = ";")
dt <- dt[!excluded_any]

# ------------------------------------------------------------
# BUILD BASKETS
# ------------------------------------------------------------

dt[, BasketID := .GRP, by = .(Person, Date, Channel)]
setkey(dt, BasketID, Product)
dt_mining <- unique(dt, by = c("BasketID", "Product"))

all_products <- unique(dt_mining$Product)
fwrite(data.table(Product = all_products), all_products_file, sep = ";")

# ------------------------------------------------------------
# CONVERT TO TRANSACTIONS
# ------------------------------------------------------------

transactions <- as(split(dt_mining$Product, dt_mining$BasketID), "transactions")
rm(dt_mining)
gc()

# ------------------------------------------------------------
# ITEM FREQUENCY DIAGNOSTICS
# ------------------------------------------------------------

item_frequency <- itemFrequency(transactions, type = "absolute")
item_frequency_df <- data.table(
  Product = names(item_frequency),
  Count = as.numeric(item_frequency)
)
setorder(item_frequency_df, -Count)
fwrite(item_frequency_df, item_frequency_file, sep = ";")

# ------------------------------------------------------------
# APRIORI MINING
# ------------------------------------------------------------

rules <- apriori(
  transactions,
  parameter = list(
    supp = MIN_SUPPORT,
    conf = MIN_CONF,
    minlen = 2,
    maxlen = MAXLEN_RULE,
    target = "rules"
  )
)

# ------------------------------------------------------------
# RULE FILTERING
# ------------------------------------------------------------

if (MIN_LIFT_KEEP > 0) {
  rules <- subset(rules, lift >= MIN_LIFT_KEEP)
}

if (REMOVE_REDUNDANT && length(rules) > 0) {
  rules <- rules[!is.redundant(rules)]
}

# ------------------------------------------------------------
# PRODUCTS PRESENT IN RULES
# ------------------------------------------------------------

if (length(rules) > 0) {
  lhs_items <- unlist(strsplit(gsub("[\\{\\}]", "", labels(lhs(rules))), ","))
  rhs_items <- unlist(strsplit(gsub("[\\{\\}]", "", labels(rhs(rules))), ","))
  products_in_rules <- unique(trimws(c(lhs_items, rhs_items)))
  fwrite(data.table(Product = products_in_rules), products_in_rules_file, sep = ";")
} else {
  fwrite(data.table(Product = character()), products_in_rules_file, sep = ";")
}

# ------------------------------------------------------------
# RULE EXPORT
# ------------------------------------------------------------

final <- data.table(
  Premises = gsub("[\\{\\}]", "", labels(lhs(rules))),
  Conclusion = gsub("[\\{\\}]", "", labels(rhs(rules))),
  Support = quality(rules)$support,
  Confidence = quality(rules)$confidence,
  Lift = quality(rules)$lift
)

fwrite(final, output_file, sep = ";")
