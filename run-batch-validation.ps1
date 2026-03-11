param(
    [string]$BaseUrl = "http://127.0.0.1:8094",
    [string]$DatasetNameLike = "*",
    [string[]]$DatasetIds = @(),
    [int]$MaxRows = 500,
    [switch]$AnalyzeAllRows,
    [switch]$IncludeSemanticsCheck
)

$ErrorActionPreference = "Stop"

function ConvertTo-NormalizedTag {
    param([object]$Value)
    if ($null -eq $Value) { return "" }
    $txt = [string]$Value
    return (($txt.ToUpper()) -replace "[^A-Z]", "")
}

function Get-ApiJson {
    param(
        [string]$Method,
        [string]$Uri,
        [object]$Body = $null
    )

    try {
        if ($Method -eq "GET") {
            return Invoke-RestMethod -Method Get -Uri $Uri -TimeoutSec 120
        }

        if ($null -ne $Body) {
            $jsonBody = $Body | ConvertTo-Json -Depth 10
            return Invoke-RestMethod -Method $Method -Uri $Uri -ContentType "application/json" -Body $jsonBody -TimeoutSec 120
        }

        return Invoke-RestMethod -Method $Method -Uri $Uri -TimeoutSec 120
    }
    catch {
        $msg = $_.Exception.Message
        throw "API call failed: $Method $Uri :: $msg"
    }
}

function Test-QuestionResult {
    param(
        [string]$TestId,
        [string]$Question,
        [object]$Response
    )

    $errors = @()

    if ($null -eq $Response) {
        $errors += "Empty response"
        return @{ Pass = $false; Errors = $errors }
    }

    $rowsReturned = 0
    if ($null -ne $Response.rows_returned) {
        $rowsReturned = [int]$Response.rows_returned
    }

    $rows = @()
    if ($null -ne $Response.rows) {
        $rows = @($Response.rows)
    }

    $routeIntent = [string]$Response.route_intent
    $routeSource = [string]$Response.route_source

    if ($rowsReturned -lt 1) {
        $errors += "rows_returned=0"
    }

    switch ($TestId) {
        "Q1" {
            # Expect deterministic unbilled summary-style routing.
            if ($routeIntent -notmatch "^(exception_summary|unbilled_summary_ops)$") {
                $errors += "unexpected route_intent=$routeIntent"
            }

            if ($rows.Count -gt 0 -and ($rows[0].PSObject.Properties.Name -contains "exception_type")) {
                $invalid = @($rows | Where-Object { (ConvertTo-NormalizedTag $_.exception_type) -ne "UNBILLED" })
                if ($invalid.Count -gt 0) {
                    $errors += "response includes non-UNBILLED exception_type values"
                }
            }
        }
        "Q2" {
            # Expect filtered billing-delay exception list.
            if ($routeIntent -ne "exception_list") {
                $errors += "unexpected route_intent=$routeIntent"
            }

            if ($rows.Count -gt 0 -and ($rows[0].PSObject.Properties.Name -contains "exception_type")) {
                $invalid = @($rows | Where-Object { (ConvertTo-NormalizedTag $_.exception_type) -ne "BILLINGDELAY" })
                if ($invalid.Count -gt 0) {
                    $errors += "response includes non-BILLING_DELAY exception_type values"
                }
            }

            if ($rows.Count -gt 0 -and ($rows[0].PSObject.Properties.Name -contains "recommended_action")) {
                $missing = @($rows | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.recommended_action) })
                if ($missing.Count -gt 0) {
                    $errors += "one or more rows missing recommended_action"
                }
            }
        }
        default {
            # No extra checks.
        }
    }

    return @{
        Pass = ($errors.Count -eq 0)
        Errors = $errors
        RouteIntent = $routeIntent
        RouteSource = $routeSource
        RowsReturned = $rowsReturned
    }
}

function Get-DatasetTableNames {
    param([object]$Dataset)

    $tables = @()
    try {
        if ($null -ne $Dataset.relationships -and $null -ne $Dataset.relationships.tables) {
            $tables = @($Dataset.relationships.tables | ForEach-Object { [string]$_ })
        }
    }
    catch {
        $tables = @()
    }
    return $tables
}

function Get-DatasetColumns {
    param([object]$Dataset)

    $columns = @()
    try {
        if ($null -ne $Dataset.columns) {
            $columns = @($Dataset.columns | ForEach-Object { [string]$_ })
        }
    }
    catch {
        $columns = @()
    }
    return $columns
}

function Test-HasColumnLike {
    param(
        [string[]]$Columns,
        [string[]]$Candidates
    )

    $normalized = @($Columns | ForEach-Object { ([string]$_).ToLower() -replace "[^a-z0-9]", "" })
    foreach ($candidate in $Candidates) {
        $candidateNorm = ([string]$candidate).ToLower() -replace "[^a-z0-9]", ""
        if ($normalized -contains $candidateNorm) {
            return $true
        }
    }
    return $false
}

function Test-IsBillingDataset {
    param([object]$Dataset)

    $tables = @(Get-DatasetTableNames -Dataset $Dataset)
    if ($tables.Count -eq 0) {
        return $false
    }

    $norm = @($tables | ForEach-Object { ([string]$_).ToLower().Replace("_", "") })

    if ($norm -contains "agentoutput") { return $true }
    if ($norm -contains "summary") { return $true }
    if (($norm -contains "opsevents") -and ($norm -contains "invoicessynthetic")) { return $true }

    return $false
}

function Test-IsSalesDataset {
    param([object]$Dataset)

    $name = [string]$Dataset.name
    $nameNorm = $name.ToLower().Replace("_", "").Replace("-", "")
    $columns = @(Get-DatasetColumns -Dataset $Dataset)

    $hasRevenueCol = Test-HasColumnLike -Columns $columns -Candidates @(
        "TotalPrice", "SalesAmount", "NetSales", "TotalAmount", "Amount"
    )
    $hasDateCol = Test-HasColumnLike -Columns $columns -Candidates @(
        "OrderDate", "Date", "InvoiceDate", "DeliveryDate", "ServiceDate"
    )
    $hasDimensionCol = Test-HasColumnLike -Columns $columns -Candidates @(
        "Product", "Region", "StoreLocation", "Salesperson", "CustomerName"
    )

    if ($nameNorm -like "*sales*" -and $hasRevenueCol -and $hasDateCol) {
        return $true
    }
    if ($hasRevenueCol -and $hasDateCol -and $hasDimensionCol) {
        return $true
    }

    return $false
}

function Get-DatasetFamily {
    param([object]$Dataset)

    if (Test-IsBillingDataset -Dataset $Dataset) {
        return "billing"
    }
    if (Test-IsSalesDataset -Dataset $Dataset) {
        return "sales"
    }
    return "other"
}

function Get-TestSuiteForDataset {
    param([object]$Dataset)

    $family = Get-DatasetFamily -Dataset $Dataset
    switch ($family) {
        "billing" {
            return @(
                @{
                    Id = "Q1"
                    Question = "how many unbilled events are there?"
                    AnalyzeAllRows = $true
                },
                @{
                    Id = "Q2"
                    Question = 'can You give the recommended action for the exception_type="Billing_Delayed".'
                    AnalyzeAllRows = $true
                }
            )
        }
        "sales" {
            return @(
                @{
                    Id = "S1"
                    Question = "Show total sales by region"
                    AnalyzeAllRows = $true
                },
                @{
                    Id = "S2"
                    Question = "Top products by revenue"
                    AnalyzeAllRows = $true
                },
                @{
                    Id = "S3"
                    Question = "which Product Region has Highest Revenue in Year 2025"
                    AnalyzeAllRows = $true
                }
            )
        }
        default {
            return @()
        }
    }
}

function Test-SalesQuestionResult {
    param(
        [string]$TestId,
        [string]$Question,
        [object]$Response
    )

    $errors = @()

    if ($null -eq $Response) {
        $errors += "Empty response"
        return @{ Pass = $false; Errors = $errors }
    }

    $rowsReturned = 0
    if ($null -ne $Response.rows_returned) {
        $rowsReturned = [int]$Response.rows_returned
    }

    $rows = @()
    if ($null -ne $Response.rows) {
        $rows = @($Response.rows)
    }

    $routeIntent = [string]$Response.route_intent
    $routeSource = [string]$Response.route_source

    if ($rowsReturned -lt 1) {
        $errors += "rows_returned=0"
    }

    switch ($TestId) {
        "S1" {
            if ($rowsReturned -lt 3) {
                $errors += "expected at least 3 region rows"
            }
            if ($rows.Count -gt 0) {
                $props = @($rows[0].PSObject.Properties.Name)
                if (-not ($props -contains "region")) {
                    $errors += "missing region column"
                }
                if (-not (($props -contains "totalprice") -or ($props -contains "total_revenue") -or ($props -contains "total_sales"))) {
                    $errors += "missing revenue metric column"
                }
            }
        }
        "S2" {
            if ($rows.Count -gt 0) {
                $props = @($rows[0].PSObject.Properties.Name)
                if (-not ($props -contains "product")) {
                    $errors += "missing product column"
                }
                if (-not (($props -contains "totalprice") -or ($props -contains "total_revenue") -or ($props -contains "total_sales"))) {
                    $errors += "missing revenue metric column"
                }
            }
        }
        "S3" {
            if ($routeIntent -ne "top_revenue_by_year") {
                $errors += "unexpected route_intent=$routeIntent"
            }
            if ($routeSource -ne "deterministic_year_revenue_fallback") {
                $errors += "unexpected route_source=$routeSource"
            }
            if ($rows.Count -gt 0) {
                $props = @($rows[0].PSObject.Properties.Name)
                foreach ($required in @("product", "region", "total_revenue")) {
                    if (-not ($props -contains $required)) {
                        $errors += "missing $required column"
                    }
                }
            }
        }
        default {
        }
    }

    return @{
        Pass = ($errors.Count -eq 0)
        Errors = $errors
        RouteIntent = $routeIntent
        RouteSource = $routeSource
        RowsReturned = $rowsReturned
    }
}

$datasets = @()
$allDatasetsRaw = Get-ApiJson -Method "GET" -Uri "$($BaseUrl.TrimEnd('/'))/datasets"
$allDatasets = @($allDatasetsRaw | ForEach-Object { $_ })

if ($DatasetIds.Count -gt 0) {
    $idSet = @{}
    foreach ($id in $DatasetIds) {
        $idSet[[string]$id] = $true
    }
    $datasets = @($allDatasets | Where-Object { $idSet.ContainsKey([string]$_.id) })
}
else {
    $datasets = @($allDatasets | Where-Object { [string]$_.name -like $DatasetNameLike })
}

if ($datasets.Count -eq 0) {
    throw "No datasets matched. DatasetNameLike='$DatasetNameLike' DatasetIds='$($DatasetIds -join ',')'"
}

$datasets = @($datasets | Sort-Object created_at -Descending)

Write-Host "Using API: $BaseUrl" -ForegroundColor Cyan
Write-Host "Datasets selected: $($datasets.Count)" -ForegroundColor Cyan

$results = @()

foreach ($dataset in $datasets) {
    $datasetId = [string]$dataset.id
    $datasetName = [string]$dataset.name
    $datasetFamily = Get-DatasetFamily -Dataset $dataset
    $tests = @(Get-TestSuiteForDataset -Dataset $dataset)

    Write-Host "" 
    Write-Host "Dataset: $datasetName ($datasetId)" -ForegroundColor Yellow
    Write-Host "  family: $datasetFamily" -ForegroundColor DarkCyan

    if ($IncludeSemanticsCheck) {
        try {
            $sem = Get-ApiJson -Method "GET" -Uri "$($BaseUrl.TrimEnd('/'))/dataset_semantics/$datasetId"
            $hasRoles = ($null -ne $sem.semantic_roles -and ($sem.semantic_roles.PSObject.Properties.Count -gt 0))
            $hasCleaning = ($null -ne $sem.cleaning_report -and ($sem.cleaning_report.PSObject.Properties.Count -gt 0))
            Write-Host "  semantics: roles=$hasRoles cleaning_report=$hasCleaning" -ForegroundColor DarkCyan
        }
        catch {
            Write-Host "  semantics check failed: $($_.Exception.Message)" -ForegroundColor DarkYellow
        }
    }

    if ($tests.Count -eq 0) {
        $tableNames = @((Get-DatasetTableNames -Dataset $dataset) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        $tableInfo = if ($tableNames.Count -gt 0) { $tableNames -join ", " } else { "none" }
        Write-Host "  No configured test suite for this dataset (tables: $tableInfo). Validation will be skipped." -ForegroundColor DarkYellow
    }

    foreach ($test in $tests) {
        $askAnalyzeAll = [bool]$test.AnalyzeAllRows
        if ($AnalyzeAllRows) {
            $askAnalyzeAll = $true
        }

        $payload = @{
            dataset_id = $datasetId
            question = [string]$test.Question
            max_rows = $MaxRows
            analyze_all_rows = $askAnalyzeAll
        }

        try {
            $response = Get-ApiJson -Method "POST" -Uri "$($BaseUrl.TrimEnd('/'))/ask_sql" -Body $payload
            if ($datasetFamily -eq "sales") {
                $check = Test-SalesQuestionResult -TestId ([string]$test.Id) -Question ([string]$test.Question) -Response $response
            }
            else {
                $check = Test-QuestionResult -TestId ([string]$test.Id) -Question ([string]$test.Question) -Response $response
            }

            $status = if ($check.Pass) { "PASS" } else { "FAIL" }
            $color = if ($check.Pass) { "Green" } else { "Red" }

            Write-Host ("  [{0}] {1} | route={2}/{3} rows={4}" -f $status, $test.Id, $check.RouteIntent, $check.RouteSource, $check.RowsReturned) -ForegroundColor $color

            if (-not $check.Pass) {
                foreach ($err in $check.Errors) {
                    Write-Host "    - $err" -ForegroundColor Red
                }
            }

            $results += [pscustomobject]@{
                dataset_id = $datasetId
                dataset_name = $datasetName
                test_id = [string]$test.Id
                question = [string]$test.Question
                pass = [bool]$check.Pass
                skipped = $false
                route_intent = [string]$check.RouteIntent
                route_source = [string]$check.RouteSource
                rows_returned = [int]$check.RowsReturned
                errors = [string](($check.Errors -join "; "))
            }
        }
        catch {
            $msg = $_.Exception.Message
            Write-Host ("  [FAIL] {0} | request error: {1}" -f $test.Id, $msg) -ForegroundColor Red
            $results += [pscustomobject]@{
                dataset_id = $datasetId
                dataset_name = $datasetName
                test_id = [string]$test.Id
                question = [string]$test.Question
                pass = $false
                skipped = $false
                route_intent = ""
                route_source = ""
                rows_returned = 0
                errors = $msg
            }
        }
    }
}

$total = $results.Count
$skipped = @($results | Where-Object { $_.skipped }).Count
$failed = @($results | Where-Object { (-not $_.pass) -and (-not $_.skipped) }).Count
$passed = @($results | Where-Object { $_.pass -and (-not $_.skipped) }).Count

Write-Host ""
Write-Host "Summary: total=$total pass=$passed fail=$failed skip=$skipped" -ForegroundColor Cyan

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "Failed cases:" -ForegroundColor Red
    $results |
        Where-Object { (-not $_.pass) -and (-not $_.skipped) } |
        Select-Object dataset_name, dataset_id, test_id, route_intent, rows_returned, errors |
        Format-Table -AutoSize | Out-String | Write-Host
    exit 1
}

exit 0
