# ============================================================
# MARKET BASKET ANALYSIS: DATA PREPARATION AND APRIORI MINING
# ============================================================

# The script keeps the manuscript baseline as its default configuration.
# Every important parameter can be overridden from the command line, so
# experiments no longer require editing the source code.

get_script_dir <- function() {
  command_args <- commandArgs(trailingOnly = FALSE)
  script_arg <- grep("^--file=", command_args, value = TRUE)

  if (length(script_arg) > 0L) {
    return(dirname(normalizePath(
      sub("^--file=", "", script_arg[[1]]),
      winslash = "/"
    )))
  }

  frame_file <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) NULL)
  if (!is.null(frame_file)) {
    return(dirname(normalizePath(frame_file, winslash = "/")))
  }

  normalizePath(getwd(), winslash = "/")
}

print_help <- function() {
  cat(
    paste(
      "Usage:",
      "  Rscript apriori.R [options]",
      "",
      "Options:",
      "  --input-dir=PATH              CSV input directory (default: Data)",
      "  --output-dir=PATH             Output directory (default: script directory)",
      "  --file-pattern=REGEX          Input filename pattern (default: ^transactions.*[.]csv$)",
      "  --support=NUMBER              Minimum support (default: 0.001)",
      "  --confidence=NUMBER           Minimum confidence (default: 0.30)",
      "  --maxlen=INTEGER              Maximum rule length (default: 3)",
      "  --maxtime=SECONDS             Apriori mining limit; 0 disables it (default: 0)",
      "  --min-lift=NUMBER             Post-mining lift filter (default: 1.0)",
      "  --remove-arules-redundant=BOOL  Apply arules redundancy filter (default: true)",
      "  --help                        Show this message",
      sep = "\n"
    ),
    "\n"
  )
}

parse_boolean <- function(value, option_name) {
  normalized <- tolower(trimws(value))
  if (normalized %in% c("true", "t", "1", "yes", "y")) {
    return(TRUE)
  }
  if (normalized %in% c("false", "f", "0", "no", "n")) {
    return(FALSE)
  }
  stop(option_name, " must be true or false.")
}

parse_cli <- function(script_dir) {
  config <- list(
    input_dir = file.path(script_dir, "Data"),
    output_dir = script_dir,
    file_pattern = "^transactions.*[.]csv$",
    min_support = 0.001,
    min_confidence = 0.30,
    max_rule_length = 3L,
    max_mining_time = 0,
    min_lift = 1.0,
    remove_arules_redundant = TRUE
  )

  args <- commandArgs(trailingOnly = TRUE)
  if ("--help" %in% args) {
    print_help()
    quit(save = "no", status = 0L)
  }

  for (arg in args) {
    if (!grepl("^--[^=]+=", arg)) {
      stop("Arguments must use --name=value syntax. Invalid argument: ", arg)
    }

    option <- sub("=.*$", "", arg)
    value <- sub("^[^=]*=", "", arg)

    if (option == "--input-dir") {
      config$input_dir <- value
    } else if (option == "--output-dir") {
      config$output_dir <- value
    } else if (option == "--file-pattern") {
      config$file_pattern <- value
    } else if (option == "--support") {
      config$min_support <- as.numeric(value)
    } else if (option == "--confidence") {
      config$min_confidence <- as.numeric(value)
    } else if (option == "--maxlen") {
      config$max_rule_length <- as.integer(value)
    } else if (option == "--maxtime") {
      config$max_mining_time <- as.numeric(value)
    } else if (option == "--min-lift") {
      config$min_lift <- as.numeric(value)
    } else if (option == "--remove-arules-redundant") {
      config$remove_arules_redundant <- parse_boolean(value, option)
    } else {
      stop("Unknown option: ", option)
    }
  }

  if (is.na(config$min_support) ||
      config$min_support <= 0 || config$min_support > 1) {
    stop("--support must be greater than 0 and at most 1.")
  }
  if (is.na(config$min_confidence) ||
      config$min_confidence <= 0 || config$min_confidence > 1) {
    stop("--confidence must be greater than 0 and at most 1.")
  }
  if (is.na(config$max_rule_length) || config$max_rule_length < 2L) {
    stop("--maxlen must be an integer greater than or equal to 2.")
  }
  if (is.na(config$max_mining_time) || config$max_mining_time < 0) {
    stop("--maxtime must be greater than or equal to 0.")
  }
  if (is.na(config$min_lift) || config$min_lift < 0) {
    stop("--min-lift must be greater than or equal to 0.")
  }

  config$input_dir <- normalizePath(
    config$input_dir,
    winslash = "/",
    mustWork = TRUE
  )
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)
  config$output_dir <- normalizePath(
    config$output_dir,
    winslash = "/",
    mustWork = TRUE
  )

  config
}

script_dir <- get_script_dir()
config <- parse_cli(script_dir)

required_packages <- c("data.table", "arules")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Missing R packages: ", paste(missing_packages, collapse = ", "),
    ". Install them before running this script."
  )
}

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(arules))
setDTthreads(0)

# Stable output names keep the Python stages simple. Experiment runners should
# use a separate output directory for each parameter configuration.
output_paths <- list(
  rules = file.path(config$output_dir, "Rules_For_Python.csv"),
  all_products = file.path(config$output_dir, "ALL_PRODUCTS.csv"),
  item_frequency = file.path(config$output_dir, "ITEM_FREQUENCY.csv"),
  products_in_rules = file.path(config$output_dir, "PRODUCTS_IN_RULES.csv"),
  excluded_rows = file.path(config$output_dir, "EXCLUDED_ROWS_SUMMARY.csv"),
  timing = file.path(config$output_dir, "APRIORI_TIMING_SUMMARY.csv"),
  metadata = file.path(config$output_dir, "APRIORI_RUN_METADATA.csv")
)

column_candidates <- list(
  person = c("PERSON_PUBLIC_KEY", "PERSON PUBLIC KEY", "Person"),
  date = c("DATE", "Date"),
  channel = c("CHANNEL", "ONLINE/OFFLINE", "Channel"),
  product = c("PRODUCT_CATEGORY", "PRODUCT CATEGORY", "Product")
)
itemset_separator <- " || "

normalize_column_name <- function(value) {
  toupper(gsub("[^A-Z0-9]+", "", value))
}

resolve_column <- function(columns, candidates, label, file) {
  positions <- match(
    normalize_column_name(candidates),
    normalize_column_name(columns)
  )
  positions <- positions[!is.na(positions)]

  if (!length(positions)) {
    stop(
      sprintf(
        "File '%s' does not contain a %s column. Available columns: %s",
        basename(file),
        label,
        paste(columns, collapse = ", ")
      )
    )
  }

  columns[positions[[1]]]
}

load_file <- function(file) {
  header <- names(fread(file, nrows = 0L, showProgress = FALSE))
  selected_columns <- c(
    resolve_column(header, column_candidates$person, "person", file),
    resolve_column(header, column_candidates$date, "date", file),
    resolve_column(header, column_candidates$channel, "channel", file),
    resolve_column(header, column_candidates$product, "product", file)
  )

  # Reading only the four required columns reduces memory use for the full data.
  data <- fread(
    file,
    fill = TRUE,
    showProgress = FALSE,
    select = selected_columns
  )
  setnames(data, selected_columns, c("Person", "Date", "Channel", "Product"))
  data
}

serialize_itemsets <- function(itemsets) {
  # A dedicated separator is necessary because product names may contain commas.
  vapply(
    LIST(itemsets, decode = TRUE),
    function(items) paste(items, collapse = itemset_separator),
    character(1)
  )
}

pipeline_start <- Sys.time()
preparation_start <- Sys.time()

input_files <- sort(list.files(
  path = config$input_dir,
  pattern = config$file_pattern,
  full.names = TRUE,
  ignore.case = TRUE
))
# Validation and summary exports may live beside a sample dataset, but they are
# metadata rather than transaction inputs and must never be mined as baskets.
input_files <- input_files[
  !grepl("_(validation|summary)[.]csv$", basename(input_files), ignore.case = TRUE)
]
if (!length(input_files)) {
  stop("No CSV files found in input directory: ", config$input_dir)
}

message("Loading ", length(input_files), " input file(s) from ", config$input_dir)
data <- rbindlist(lapply(input_files, load_file), use.names = TRUE)
rows_before_cleaning <- nrow(data)

data[, Person := trimws(as.character(Person))]
data[, Date := trimws(as.character(Date))]
data[, Channel := toupper(trimws(as.character(Channel)))]
data[, Product := trimws(as.character(Product))]

# A missing channel is retained as UNKNOWN so an otherwise valid basket can
# still be formed. Rows without person, date, or product cannot define a basket.
data[is.na(Channel) | Channel == "", Channel := "UNKNOWN"]
missing_person <- is.na(data$Person) | data$Person == ""
missing_date <- is.na(data$Date) | data$Date == ""
missing_product <- is.na(data$Product) | data$Product == ""
excluded_row <- missing_person | missing_date | missing_product

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
    rows_before_cleaning,
    sum(missing_person),
    sum(missing_date),
    sum(missing_product),
    sum(excluded_row),
    sum(!excluded_row)
  )
)
fwrite(excluded_summary, output_paths$excluded_rows, sep = ";")
data <- data[!excluded_row]
rows_after_cleaning <- nrow(data)

# The manuscript defines one basket by person, date, and sales channel.
data[, BasketID := .GRP, by = .(Person, Date, Channel)]
number_of_baskets <- uniqueN(data$BasketID)

# Association-rule transactions are sets, so duplicate products inside the
# same basket are intentionally collapsed before conversion to arules format.
setkey(data, BasketID, Product)
mining_data <- unique(data, by = c("BasketID", "Product"))

all_products <- sort(unique(mining_data$Product))
fwrite(data.table(Product = all_products), output_paths$all_products, sep = ";")

transactions <- as(split(mining_data$Product, mining_data$BasketID), "transactions")
rm(mining_data, data)
invisible(gc())

item_frequency <- data.table(
  Product = names(itemFrequency(transactions, type = "absolute")),
  Count = as.numeric(itemFrequency(transactions, type = "absolute"))
)
setorder(item_frequency, -Count, Product)
fwrite(item_frequency, output_paths$item_frequency, sep = ";")

preparation_end <- Sys.time()
mining_start <- Sys.time()

message(
  "Running Apriori: support=", config$min_support,
  ", confidence=", config$min_confidence,
  ", maxlen=", config$max_rule_length,
  ", maxtime=", config$max_mining_time
)
rules <- apriori(
  transactions,
  parameter = list(
    supp = config$min_support,
    conf = config$min_confidence,
    minlen = 2L,
    maxlen = config$max_rule_length,
    # arules otherwise stops after five seconds, which can silently truncate
    # longer-rule experiments before the requested maxlen has been evaluated.
    maxtime = config$max_mining_time,
    target = "rules"
  )
)
rules_after_apriori <- length(rules)
mining_end <- Sys.time()

# Lift filtering removes associations that do not exceed the selected strength
# threshold. With the manuscript baseline, lift must be at least 1.
if (config$min_lift > 0 && length(rules) > 0L) {
  rules <- subset(rules, lift >= config$min_lift)
}
rules_after_lift <- length(rules)

# This is arules' standard redundancy filter, applied before the study's custom
# confidence-improvement pruning in Python. It remains enabled by default to
# preserve the established analysis, but is explicit and can be disabled.
if (config$remove_arules_redundant && length(rules) > 0L) {
  rules <- rules[!is.redundant(rules)]
}
rules_after_arules_redundancy <- length(rules)

if (length(rules) > 0L) {
  products_in_rules <- sort(unique(c(
    unlist(LIST(lhs(rules), decode = TRUE), use.names = FALSE),
    unlist(LIST(rhs(rules), decode = TRUE), use.names = FALSE)
  )))
} else {
  products_in_rules <- character()
}
fwrite(
  data.table(Product = products_in_rules),
  output_paths$products_in_rules,
  sep = ";"
)

exported_rules <- data.table(
  Premises = serialize_itemsets(lhs(rules)),
  Conclusion = serialize_itemsets(rhs(rules)),
  Support = quality(rules)$support,
  Confidence = quality(rules)$confidence,
  Lift = quality(rules)$lift
)
fwrite(exported_rules, output_paths$rules, sep = ";")

pipeline_end <- Sys.time()
elapsed_seconds <- function(end, start) {
  round(as.numeric(difftime(end, start, units = "secs")), 2)
}

timing_summary <- data.table(
  Step = c(
    "Data preparation",
    "Apriori mining",
    "Post-Apriori filtering and export",
    "Total pipeline"
  ),
  Seconds = c(
    elapsed_seconds(preparation_end, preparation_start),
    elapsed_seconds(mining_end, mining_start),
    elapsed_seconds(pipeline_end, mining_end),
    elapsed_seconds(pipeline_end, pipeline_start)
  )
)
fwrite(timing_summary, output_paths$timing, sep = ";")

metadata <- data.table(
  Metric = c(
    "Input directory",
    "Input file pattern",
    "Input files",
    "Rows before cleaning",
    "Rows after cleaning",
    "Baskets",
    "Product categories",
    "Minimum support",
    "Minimum confidence",
    "Minimum lift",
    "Maximum rule length",
    "Maximum Apriori mining time (seconds)",
    "Remove arules redundant rules",
    "Rules after Apriori",
    "Rules after lift filter",
    "Rules exported",
    "R version",
    "data.table version",
    "arules version"
  ),
  Value = as.character(c(
    config$input_dir,
    config$file_pattern,
    length(input_files),
    rows_before_cleaning,
    rows_after_cleaning,
    number_of_baskets,
    length(all_products),
    config$min_support,
    config$min_confidence,
    config$min_lift,
    config$max_rule_length,
    config$max_mining_time,
    config$remove_arules_redundant,
    rules_after_apriori,
    rules_after_lift,
    rules_after_arules_redundancy,
    paste(R.version$major, R.version$minor, sep = "."),
    as.character(packageVersion("data.table")),
    as.character(packageVersion("arules"))
  ))
)
fwrite(metadata, output_paths$metadata, sep = ";")

cat("\nApriori stage completed.\n")
print(metadata[Metric %in% c(
  "Rows after cleaning",
  "Baskets",
  "Product categories",
  "Rules after Apriori",
  "Rules after lift filter",
  "Rules exported"
)])
print(timing_summary)
cat("Rules saved to:", output_paths$rules, "\n")
