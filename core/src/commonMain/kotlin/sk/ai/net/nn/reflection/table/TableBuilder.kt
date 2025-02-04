package sk.ai.net.nn.reflection.table

// DSL entry point
fun table(block: Table.() -> Unit): Table {
    return Table().apply(block)
}

// The Table DSL class
class Table {
    // A cell style configuration object
    val cellStyle = CellStyle()

    // Optional header section
    var header: Header? = null

    // List of body rows
    val rows = mutableListOf<Row>()

    // DSL function to configure the cell style
    fun cellStyle(block: CellStyle.() -> Unit) {
        cellStyle.block()
    }

    // DSL function to add a header section
    fun header(block: Header.() -> Unit) {
        header = Header().apply(block)
    }

    // DSL function to add a body row directly to the table
    fun row(block: Row.() -> Unit) {
        rows.add(Row().apply(block))
    }

    // Converts the table to an ASCII string
    override fun toString(): String {
        // Gather all rows (header and body) to compute column widths.
        val allRows = mutableListOf<Row>()
        header?.let { allRows.addAll(it.rows) }
        allRows.addAll(rows)

        // Determine the number of columns (max among all rows).
        val colCount = allRows.maxOfOrNull { it.cells.size } ?: 0
        val colWidths = MutableList(colCount) { 0 }
        for (row in allRows) {
            row.cells.forEachIndexed { index, cell ->
                colWidths[index] = maxOf(colWidths[index], cell.content.length)
            }
        }

        // Build the table line by line.
        val sb = StringBuilder()

        // If borders are enabled, print a top border.
        if (cellStyle.border) {
            sb.appendLine(buildSeparatorLine(colWidths))
        }

        // Print header rows if they exist.
        header?.let {
            for (row in it.rows) {
                sb.appendLine(buildRowLine(row, colWidths, cellStyle.border))
            }
            if (cellStyle.border) {
                sb.appendLine(buildSeparatorLine(colWidths))
            }
        }

        // Print body rows.
        for (row in rows) {
            sb.appendLine(buildRowLine(row, colWidths, cellStyle.border))
            if (cellStyle.border) {
                sb.appendLine(buildSeparatorLine(colWidths))
            }
        }

        return sb.toString()
    }

    // Helper: builds a border/separator line based on column widths.
    private fun buildSeparatorLine(colWidths: List<Int>): String {
        return colWidths.joinToString(separator = "+", prefix = "+", postfix = "+") {
            "-".repeat(it + 2)
        }
    }

    // Helper: builds a formatted row line.
    private fun buildRowLine(row: Row, colWidths: List<Int>, border: Boolean): String {
        val cells = row.cells.mapIndexed { index, cell ->
            " " + cell.content.padEnd(colWidths[index]) + " "
        }
        return if (border) {
            cells.joinToString(separator = "|", prefix = "|", postfix = "|")
        } else {
            cells.joinToString(separator = " ")
        }
    }
}

// A simple header container allowing multiple header rows.
class Header {
    val rows = mutableListOf<Row>()

    fun row(block: Row.() -> Unit) {
        rows.add(Row().apply(block))
    }
}

// Represents a row in the table.
class Row {
    val cells = mutableListOf<Cell>()

    // Adds a cell to the row.
    fun cell(value: Any?) {
        cells.add(Cell(value?.toString() ?: ""))
    }
}

// Represents a cell containing text.
class Cell(val content: String)

// A configuration class for cell style options.
class CellStyle {
    var border: Boolean = false
}
