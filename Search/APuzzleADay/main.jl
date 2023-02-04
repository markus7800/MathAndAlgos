import ProgressLogging: @progress

A = BitMatrix([
    1 1 0;
    1 1 1
])

function roto_flip(piece)
    flipped_piece = Matrix(piece')
    return Set(BitMatrix[
        piece, rotr90(piece), rotl90(piece), rot180(piece),
        flipped_piece, rotr90(flipped_piece), rotl90(flipped_piece), rot180(flipped_piece),
    ])
end

roto_flip(A)

const PIECES = [
    [
        1 1 0;
        1 1 1
    ],
    [
        1 1 1;
        1 1 1
    ],
    [
        1 0 1;
        1 1 1
    ],
    [
        1 0 0;
        1 0 0;
        1 1 1
    ],
    [
        0 1 1;
        0 1 0;
        1 1 0;
    ],
    [
        0 1 1 1;
        1 1 0 0
    ],
    [
        0 0 1 0;
        1 1 1 1
    ],
    [
        1 1 1 1;
        0 0 0 1
    ]
];

const ROTO_FLIPS = roto_flip.(PIECES);
length.(ROTO_FLIPS)

for p in ROTO_FLIPS
    for m in p
        display(m)
    end
end

const BOARD = BitMatrix([
    1 1 1 1 1 1 0 0 0 0;
    1 1 1 1 1 1 0 0 0 0;
    1 1 1 1 1 1 1 0 0 0;
    1 1 1 1 1 1 1 0 0 0;
    1 1 1 1 1 1 1 0 0 0;
    1 1 1 1 1 1 1 0 0 0;
    1 1 1 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0
])

function get_mask(i::Int, j::Int)
    mask = trues(10, 10)
    mask[i, j] = false
    return mask
end

const JAN = get_mask(1, 1);
const FEB = get_mask(1, 2);
const MAR = get_mask(1, 3);
const APR = get_mask(1, 4);
const MAY = get_mask(1, 5);
const JUN = get_mask(1, 6);
const JUL = get_mask(2, 1);
const AUG = get_mask(2, 2);
const SEP = get_mask(2, 3);
const OCT = get_mask(2, 4);
const NOV = get_mask(2, 5);
const DEC = get_mask(2, 6);

function get_day_mask(day::Int)
    col = (day-1) % 7 + 1
    row = (day-1) ÷ 7 + 3
    return get_mask(row, col)
end

function get_puzzle(month, day)
    return BOARD .& month .& get_day_mask(day)
end


function is_move(puzzle, piece, row, col)
    height, width = size(piece)
    return !any(.!puzzle[row:row+height-1, col:col+width-1] .& piece)
end

function do_move!(puzzle, piece, row, col)
    height, width = size(piece)
    s = sum(puzzle)
    puzzle[row:row+height-1, col:col+width-1] .= puzzle[row:row+height-1, col:col+width-1] .& .!piece
    @assert sum(puzzle) == s - sum(piece)
    return puzzle
end

function undo_move!(puzzle, piece, row, col)
    height, width = size(piece)
    puzzle[row:row+height-1, col:col+width-1] .= puzzle[row:row+height-1, col:col+width-1] .| piece
    return puzzle
end


puzzle = get_puzzle(FEB, 14)
is_move(puzzle, A, 2, 2)
do_move!(puzzle, A, 2, 2)
undo_move!(puzzle, A, 2, 2)

function print_solution(solution)
    board = fill('*', 7, 7)
    for (i, letter) in enumerate(['A','B','C','D','E','F','G', 'H'])
        p, row, col = solution[i]
        height, width = size(p)
        for r in 1:height, c in 1:width
            if p[r,c]
                board[row+r-1,col+c-1] = letter
            end            
        end
    end
    for i in 1:7
        for j in 1:7
            print(board[i, j], " ")
        end
        println()
    end

end


function backtrack(puzzle, pieces)
    if isempty(pieces)
        return true, Dict()
    end

    for x in 1:7, y in 1:7
        if puzzle[x, y]
            for piece in pieces
                delete!(pieces, piece)
                for p in ROTO_FLIPS[piece]
                    height, width = size(p)
                    for row in max(1,x-height):x, col in max(1,y-width):y
                        if is_move(puzzle, p, row, col)
                            puzzle_prev = copy(puzzle)
                            do_move!(puzzle, p, row, col)
                            result, solution = backtrack(puzzle, pieces)
                            if result
                                solution[piece] = (p, row, col)
                                return true, solution
                            end
                            undo_move!(puzzle, p, row, col)
                            @assert puzzle_prev == puzzle
                        end
                    end
                end
                push!(pieces, piece)
            end
            break
        end
    end
    return false, nothing
end

puzzle = get_puzzle(FEB, 4)

@time result, solution = backtrack(puzzle, Set([1,2,3,4,5,6,7,8]))

print_solution(solution)

MONTHS = [JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC];
@time @progress for month in 1:12, day in 1:31
    puzzle = get_puzzle(MONTHS[month], day)
    result, _ = backtrack(puzzle, Set([1,2,3,4,5,6,7,8]))
    if !result
        println(month, " ", day)
    end
end


function backtrack_all(puzzle, pieces, solutions, ass=Dict())
    if isempty(pieces)
        push!(solutions, copy(ass))
        return
    end

    for x in 1:7, y in 1:7
        if puzzle[x, y]
            for piece in pieces
                delete!(pieces, piece)
                for p in ROTO_FLIPS[piece]
                    height, width = size(p)
                    for row in max(1,x-height):x, col in max(1,y-width):y
                        if is_move(puzzle, p, row, col)
                            puzzle_prev = copy(puzzle)
                            do_move!(puzzle, p, row, col)
                            ass[piece] = (p, row, col)
                            backtrack_all(puzzle, pieces, solutions, ass)
                            delete!(ass, piece)
                            undo_move!(puzzle, p, row, col)
                            @assert puzzle_prev == puzzle
                        end
                    end
                end
                push!(pieces, piece)
            end
            break
        end
    end
end

puzzle = get_puzzle(OCT, 6)
solutions = []
backtrack_all(puzzle, Set([1,2,3,4,5,6,7,8]), solutions)

print_solution(solutions[1])

month_day = [];
n_solutions = [];
@progress for month in 1:12, day in 1:31
    puzzle = get_puzzle(MONTHS[month], day)
    solutions = []
    backtrack_all(puzzle, Set([1,2,3,4,5,6,7,8]), solutions)
    push!(month_day, (month, day))
    push!(n_solutions, length(solutions))
end;
n_solutions

for i in 0:maximum(n_solutions)
    println(i, ": ", sum(n_solutions .== i))
end

d = Dict(md => n for (md, n) in zip(month_day, n_solutions));

d[(2,14)]

month_day[n_solutions .== 1]