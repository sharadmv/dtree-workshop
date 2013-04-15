function[ret] = split(c, question)
    ret = struct();
    left = struct();
    right = struct();
    select = eval(question);
    ret.left = struct("x", c.x(select, :), "y", c.y(select, :));
    ret.right = struct("x", c.x(!select, :), "y", c.y(!select, :));
endfunction

function[splits] = getsplits(questions)
    num_questions = length(questions);

    splits = cell(num_questions, 1);

    for i = 1:num_questions
        splits{i} = @(c) split(c, questions{i});
    endfor
endfunction
