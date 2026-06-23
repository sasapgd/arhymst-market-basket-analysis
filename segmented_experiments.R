# ============================================================
# SEGMENTED EXPERIMENTS -> Gender, Age, and Gender x Age
# ============================================================

library(data.table)
library(arules)

setDTthreads(0)

# ------------------------------------------------------------
# COMMAND LINE AND SETTINGS
# ------------------------------------------------------------

command_args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", command_args, value = TRUE)
script_file <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "segmented_experiments.R"
BASE_DIR <- normalizePath(dirname(script_file), winslash = "/", mustWork = TRUE)

parse_args <- function(args) {
  config <- list(
    input_dir = file.path(BASE_DIR, "Data"),
    output_dir = file.path(BASE_DIR, "segmented_experiments"),
    support = 0.001,
    confidence = 0.30,
    maxlen = 3L,
    min_lift = 1.0,
    delta = 0.05,
    baseline_mst = file.path(BASE_DIR, "timing_runs", "mst_variants", "MST_MAXLEN_3_CONFIDENCE.csv")
  )
  for (arg in args) {
    pieces <- strsplit(arg, "=", fixed = TRUE)[[1]]
    if (length(pieces) != 2L) stop("Arguments must use --name=value syntax: ", arg)
    key <- pieces[1]
    value <- pieces[2]
    if (key == "--input-dir") config$input_dir <- value
    else if (key == "--output-dir") config$output_dir <- value
    else if (key == "--support") config$support <- as.numeric(value)
    else if (key == "--confidence") config$confidence <- as.numeric(value)
    else if (key == "--maxlen") config$maxlen <- as.integer(value)
    else if (key == "--min-lift") config$min_lift <- as.numeric(value)
    else if (key == "--delta") config$delta <- as.numeric(value)
    else if (key == "--baseline-mst") config$baseline_mst <- value
    else stop("Unknown argument: ", key)
  }
  if (!dir.exists(config$input_dir)) stop("Input directory not found: ", config$input_dir)
  if (!file.exists(config$baseline_mst)) stop("Baseline MaxST not found: ", config$baseline_mst)
  if (is.na(config$support) || config$support <= 0 || config$support > 1) stop("support must be in (0, 1].")
  if (is.na(config$confidence) || config$confidence <= 0 || config$confidence > 1) stop("confidence must be in (0, 1].")
  if (is.na(config$delta) || config$delta < 0 || config$delta > 1) stop("delta must be in [0, 1].")
  if (is.na(config$maxlen) || config$maxlen < 2L) stop("maxlen must be at least 2.")
  config
}

config <- parse_args(commandArgs(trailingOnly = TRUE))
INPUT_DIR <- normalizePath(config$input_dir, winslash = "/", mustWork = TRUE)
OUTPUT_DIR <- normalizePath(config$output_dir, winslash = "/", mustWork = FALSE)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

MIN_SUPPORT <- config$support
MIN_CONF <- config$confidence
MAXLEN_RULE <- config$maxlen
MIN_LIFT_KEEP <- config$min_lift
IMPROVEMENT_THRESHOLD <- config$delta
ITEMSET_SEPARATOR <- " || "
BASELINE_MST_FILE <- normalizePath(config$baseline_mst, winslash = "/", mustWork = TRUE)
SUMMARY_FILE <- file.path(OUTPUT_DIR, "SEGMENTED_EXPERIMENT_SUMMARY.csv")
DETAIL_FILE <- file.path(OUTPUT_DIR, "SEGMENTED_EXPERIMENT_TIMING.csv")

COL_PERSON <- c("PERSON_PUBLIC_KEY", "PERSON PUBLIC KEY", "Person")
COL_DATE <- c("DATE", "Date")
COL_CHANNEL <- c("CHANNEL", "ONLINE/OFFLINE", "Channel")
COL_PRODUCT <- c("PRODUCT_CATEGORY", "PRODUCT CATEGORY", "Product")
COL_GENDER <- c("GENDER", "Gender")
COL_AGE <- c("AGE", "Age")
COL_AGE_GROUP <- c("AGE_GROUP", "AGE GROUP", "AgeGroup")

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

normalize_name <- function(x) toupper(gsub("[^A-Z0-9]+", "", x))

resolve_col <- function(columns, candidates, label, file) {
  idx <- match(normalize_name(candidates), normalize_name(columns))
  idx <- idx[!is.na(idx)]

  if (!length(idx)) {
    stop(sprintf(
      "File '%s' does not contain %s. Available columns: %s",
      basename(file), label, paste(columns, collapse = ", ")
    ))
  }

  columns[idx[1]]
}

resolve_col_optional <- function(columns, candidates) {
  idx <- match(normalize_name(candidates), normalize_name(columns))
  idx <- idx[!is.na(idx)]
  if (length(idx)) columns[idx[1]] else NA_character_
}

split_items <- function(value) {
  items <- trimws(strsplit(gsub("[{}]", "", value), "\\|\\|")[[1]])
  items[items != ""]
}

serialize_itemsets <- function(itemsets) {
  vapply(
    LIST(itemsets, decode = TRUE),
    function(items) paste(items, collapse = ITEMSET_SEPARATOR),
    character(1)
  )
}

edge_key <- function(a, b) paste(sort(c(a, b)), collapse = "||")

read_baseline_edges <- function(path) {
  mst <- fread(path, sep = ";", showProgress = FALSE)
  unique(mapply(edge_key, mst$Product_1, mst$Product_2))
}

proper_subsets <- function(items) {
  if (length(items) <= 1) return(list())

  result <- list()
  idx <- 1L
  for (k in seq_len(length(items) - 1L)) {
    combos <- combn(items, k, simplify = FALSE)
    for (combo in combos) {
      result[[idx]] <- paste(sort(combo), collapse = "||")
      idx <- idx + 1L
    }
  }
  result
}

reduce_rules_confidence <- function(rules_dt) {
  if (!nrow(rules_dt)) return(rules_dt)

  rules_dt[, RuleID := .I]
  rules_dt[, PremiseKey := vapply(Premises, function(x) paste(sort(split_items(x)), collapse = "||"), character(1))]
  rules_dt[, PremiseLen := lengths(strsplit(PremiseKey, "\\|\\|"))]

  remove_ids <- integer()

  for (rhs in unique(rules_dt$Conclusion)) {
    group <- rules_dt[Conclusion == rhs][order(PremiseLen)]
    kept_by_size <- list()

    for (i in seq_len(nrow(group))) {
      row <- group[i]
      removable <- FALSE
      premise_items <- split_items(row$Premises)

      if (length(premise_items) > 1) {
        for (subset_key in proper_subsets(premise_items)) {
          parent_score <- NULL
          subset_size <- length(strsplit(subset_key, "\\|\\|")[[1]])
          bucket <- kept_by_size[[as.character(subset_size)]]

          if (!is.null(bucket) && !is.null(bucket[[subset_key]])) {
            parent_score <- bucket[[subset_key]]
          }

          if (!is.null(parent_score)) {
            improvement <- row$Confidence - parent_score
            if (improvement < IMPROVEMENT_THRESHOLD) {
              remove_ids <- c(remove_ids, row$RuleID)
              removable <- TRUE
              break
            }
          }
        }
      }

      if (!removable) {
        size_key <- as.character(row$PremiseLen)
        if (is.null(kept_by_size[[size_key]])) kept_by_size[[size_key]] <- list()
        existing <- kept_by_size[[size_key]][[row$PremiseKey]]
        if (is.null(existing) || row$Confidence > existing) {
          kept_by_size[[size_key]][[row$PremiseKey]] <- row$Confidence
        }
      }
    }
  }

  reduced <- rules_dt[!RuleID %in% remove_ids]
  reduced[, c("RuleID", "PremiseKey", "PremiseLen") := NULL]
  reduced
}

build_edges <- function(rules_dt) {
  edges <- data.table(EdgeOrder = integer(), Product_1 = character(), Product_2 = character(), Lift = numeric(), Confidence = numeric(), Weight_Lift_x_Confidence = numeric())
  if (!nrow(rules_dt)) return(edges)

  edge_map <- new.env(parent = emptyenv())
  next_edge_order <- 0L

  for (i in seq_len(nrow(rules_dt))) {
    premises <- split_items(rules_dt$Premises[i])
    conclusions <- split_items(rules_dt$Conclusion[i])
    lift <- rules_dt$Lift[i]
    confidence <- rules_dt$Confidence[i]
    weight <- lift * confidence

    for (p in premises) {
      for (c in conclusions) {
        if (p == c) next
        key <- edge_key(p, c)
        existing <- edge_map[[key]]
        if (is.null(existing) || existing$Weight_Lift_x_Confidence < weight) {
          pair <- sort(c(p, c))
          edge_order <- if (is.null(existing)) {
            value <- next_edge_order
            next_edge_order <- next_edge_order + 1L
            value
          } else existing$EdgeOrder
          edge_map[[key]] <- list(
            EdgeOrder = edge_order,
            Product_1 = pair[1],
            Product_2 = pair[2],
            Lift = lift,
            Confidence = confidence,
            Weight_Lift_x_Confidence = weight
          )
        }
      }
    }
  }

  values <- as.list(edge_map)
  if (!length(values)) return(edges)
  rbindlist(lapply(values, as.data.table), use.names = TRUE)
}

maximum_spanning_tree <- function(edges) {
  if (!nrow(edges)) return(edges)

  # EdgeOrder reproduces the stable tie behaviour of the Python main pipeline.
  edges <- copy(edges)[order(-Weight_Lift_x_Confidence, EdgeOrder)]
  nodes <- sort(unique(c(edges$Product_1, edges$Product_2)))
  parent <- setNames(nodes, nodes)
  rank <- setNames(rep(0L, length(nodes)), nodes)

  find_root <- function(x) {
    while (parent[[x]] != x) {
      parent[[x]] <<- parent[[parent[[x]]]]
      x <- parent[[x]]
    }
    x
  }

  union_nodes <- function(a, b) {
    ra <- find_root(a)
    rb <- find_root(b)
    if (ra == rb) return(FALSE)

    if (rank[[ra]] < rank[[rb]]) {
      parent[[ra]] <<- rb
    } else if (rank[[ra]] > rank[[rb]]) {
      parent[[rb]] <<- ra
    } else {
      parent[[rb]] <<- ra
      rank[[ra]] <<- rank[[ra]] + 1L
    }
    TRUE
  }

  selected <- vector("list", 0L)
  for (i in seq_len(nrow(edges))) {
    row <- edges[i]
    if (union_nodes(row$Product_1, row$Product_2)) {
      selected[[length(selected) + 1L]] <- row
    }
  }

  if (!length(selected)) return(edges[0])
  result <- rbindlist(selected, use.names = TRUE)
  node_count <- uniqueN(c(result$Product_1, result$Product_2))
  if (nrow(result) != node_count - 1L) {
    stop("Segment product graph is disconnected; a single spanning tree does not exist.")
  }
  result[, EdgeOrder := NULL]
  result
}

top_hub <- function(mst) {
  if (!nrow(mst)) return(NA_character_)
  degrees <- sort(table(c(mst$Product_1, mst$Product_2)), decreasing = TRUE)
  names(degrees)[1]
}

run_segment <- function(segment_type, segment_name, segment_dt, baseline_edges) {
  segment_start <- Sys.time()

  if (!nrow(segment_dt)) {
    return(data.table(
      SegmentType = segment_type, Segment = segment_name, Rows = 0L, Baskets = 0L,
      InitialRules = 0L, RetainedRules = 0L, MaxSTNodes = 0L, MaxSTEdges = 0L,
      SharedEdgesWithBaseline = 0L, TopHub = NA_character_, Seconds = 0
    ))
  }

  segment_dt <- copy(segment_dt)
  segment_dt[, BasketID := .GRP, by = .(Person, Date, Channel)]
  setkey(segment_dt, BasketID, Product)
  dt_mining <- unique(segment_dt, by = c("BasketID", "Product"))
  basket_count <- uniqueN(dt_mining$BasketID)

  transactions <- as(split(dt_mining$Product, dt_mining$BasketID), "transactions")

  rules <- apriori(
    transactions,
    parameter = list(
      supp = MIN_SUPPORT,
      conf = MIN_CONF,
      minlen = 2,
      maxlen = MAXLEN_RULE,
      target = "rules"
    ),
    control = list(verbose = FALSE)
  )

  if (MIN_LIFT_KEEP > 0 && length(rules) > 0) {
    rules <- subset(rules, lift >= MIN_LIFT_KEEP)
  }

  if (length(rules) > 0) {
    rules <- rules[!is.redundant(rules)]
  }

  initial_rules <- length(rules)

  if (initial_rules == 0L) {
    retained_rules <- 0L
    mst <- data.table()
    shared_edges <- 0L
    hub <- NA_character_
  } else {
    rules_dt <- data.table(
      Premises = serialize_itemsets(lhs(rules)),
      Conclusion = serialize_itemsets(rhs(rules)),
      Support = quality(rules)$support,
      Confidence = quality(rules)$confidence,
      Lift = quality(rules)$lift
    )

    reduced <- reduce_rules_confidence(rules_dt)
    retained_rules <- nrow(reduced)

    edges <- build_edges(reduced)
    mst <- maximum_spanning_tree(edges)
    mst_file <- file.path(OUTPUT_DIR, paste0(gsub("[^A-Za-z0-9]+", "_", segment_type), "__", gsub("[^A-Za-z0-9]+", "_", segment_name), "__MST.csv"))
    fwrite(mst, mst_file, sep = ";")

    mst_edges <- if (nrow(mst)) unique(mapply(edge_key, mst$Product_1, mst$Product_2)) else character()
    shared_edges <- length(intersect(mst_edges, baseline_edges))
    hub <- top_hub(mst)
  }

  seconds <- as.numeric(difftime(Sys.time(), segment_start, units = "secs"))

  data.table(
    SegmentType = segment_type,
    Segment = segment_name,
    Rows = nrow(segment_dt),
    Baskets = basket_count,
    InitialRules = initial_rules,
    RetainedRules = retained_rules,
    MaxSTNodes = if (exists("mst") && nrow(mst)) uniqueN(c(mst$Product_1, mst$Product_2)) else 0L,
    MaxSTEdges = if (exists("mst")) nrow(mst) else 0L,
    SharedEdgesWithBaseline = shared_edges,
    TopHub = hub,
    Seconds = round(seconds, 2)
  )
}

# ------------------------------------------------------------
# LOAD ORIGINAL DATA ONCE
# ------------------------------------------------------------

files <- list.files(
  INPUT_DIR,
  pattern = "^transactions.*\\.csv$",
  full.names = TRUE,
  ignore.case = TRUE
)
if (!length(files)) stop("No CSV files found.")

load_file <- function(file) {
  header <- names(fread(file, nrows = 0, showProgress = FALSE))
  person_col <- resolve_col(header, COL_PERSON, "person", file)
  date_col <- resolve_col(header, COL_DATE, "date", file)
  channel_col <- resolve_col(header, COL_CHANNEL, "channel", file)
  product_col <- resolve_col(header, COL_PRODUCT, "product", file)
  gender_col <- resolve_col(header, COL_GENDER, "gender", file)
  age_group_col <- resolve_col_optional(header, COL_AGE_GROUP)
  age_col <- resolve_col_optional(header, COL_AGE)
  if (is.na(age_group_col) && is.na(age_col)) {
    stop(sprintf(
      "File '%s' must contain either AGE_GROUP or AGE for segment analysis.",
      basename(file)
    ))
  }
  demographic_age_col <- if (!is.na(age_group_col)) age_group_col else age_col

  df <- fread(
    file,
    fill = TRUE,
    showProgress = FALSE,
    select = c(person_col, date_col, channel_col, product_col, gender_col, demographic_age_col)
  )

  setnames(
    df,
    c(person_col, date_col, channel_col, product_col, gender_col, demographic_age_col),
    c("Person", "Date", "Channel", "Product", "Gender", "AgeValue")
  )
  df[, AgeIsGrouped := !is.na(age_group_col)]
  df
}

pipeline_start <- Sys.time()
dt <- rbindlist(lapply(files, load_file), use.names = TRUE)

for (col in names(dt)) dt[, (col) := trimws(as.character(get(col)))]
dt[, Channel := toupper(Channel)]
dt[is.na(Channel) | Channel == "", Channel := "UNKNOWN"]
dt[, Gender := toupper(Gender)]
dt[Gender %in% c("Z", "Ž", "FEMALE", "WOMAN"), Gender := "F"]
dt[Gender %in% c("MALE", "MAN"), Gender := "M"]
dt[, AgeGroup := NA_character_]
dt[AgeIsGrouped == TRUE, AgeGroup := gsub("[–—]", "-", AgeValue)]
dt[AgeIsGrouped == FALSE, AgeGroup := as.character(cut(
  suppressWarnings(as.numeric(gsub(",", ".", AgeValue))),
  breaks = c(-Inf, 24, 34, 44, 54, 64, Inf),
  labels = c("18-24", "25-34", "35-44", "45-54", "55-64", "65+"),
  right = TRUE
))]

dt[, valid_row := !(is.na(Person) | Person == "" | is.na(Date) | Date == "" | is.na(Product) | Product == "")]
dt <- dt[valid_row == TRUE]
dt[, valid_row := NULL]
valid_age_groups <- c("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
dt <- dt[Gender %in% c("F", "M") & AgeGroup %in% valid_age_groups]
dt <- dt[, .(Person, Date, Channel, Product, Gender, AgeGroup)]

baseline_edges <- read_baseline_edges(BASELINE_MST_FILE)

segments <- list()
segments[[length(segments) + 1L]] <- list(type = "Gender", name = "F", data = dt[Gender == "F"])
segments[[length(segments) + 1L]] <- list(type = "Gender", name = "M", data = dt[Gender == "M"])

age_groups <- valid_age_groups
for (age_group in age_groups) {
  segments[[length(segments) + 1L]] <- list(type = "Age", name = age_group, data = dt[AgeGroup == age_group])
}

for (gender in c("F", "M")) {
  for (age_group in age_groups) {
    segments[[length(segments) + 1L]] <- list(type = "GenderAge", name = paste(gender, age_group, sep = "_"), data = dt[Gender == gender & AgeGroup == age_group])
  }
}

results <- rbindlist(lapply(segments, function(segment) {
  cat("Running segment:", segment$type, segment$name, "rows=", nrow(segment$data), "\n")
  run_segment(segment$type, segment$name, segment$data, baseline_edges)
}), use.names = TRUE, fill = TRUE)

fwrite(results, SUMMARY_FILE, sep = ";")

timing <- data.table(
  Step = c("Total segmented experiment"),
  Seconds = round(as.numeric(difftime(Sys.time(), pipeline_start, units = "secs")), 2)
)
fwrite(timing, DETAIL_FILE, sep = ";")

metadata <- data.table(
  Metric = c(
    "Input directory", "Baseline MaxST", "Minimum support", "Minimum confidence",
    "Maximum rule length", "Minimum lift", "Delta", "Rows with valid demographics",
    "Segments", "R version", "data.table version", "arules version"
  ),
  Value = c(
    INPUT_DIR, BASELINE_MST_FILE, MIN_SUPPORT, MIN_CONF, MAXLEN_RULE, MIN_LIFT_KEEP,
    IMPROVEMENT_THRESHOLD, nrow(dt), length(segments), R.version.string,
    as.character(packageVersion("data.table")), as.character(packageVersion("arules"))
  )
)
fwrite(metadata, file.path(OUTPUT_DIR, "SEGMENTED_EXPERIMENT_METADATA.csv"), sep = ";")

print(results)
print(timing)
