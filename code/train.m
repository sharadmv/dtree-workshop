source('questions.m');
source('impurity.m');

questions = cellstr([
    "c.x(:, 1) > 0"
  ; "c.x(:, 1) > 1"
  ; "c.x(:, 1) > 2"
  ; "c.x(:, 1) > 3"
  ; "c.x(:, 1) > 4"
]);

function[data] = preprocess_train(d)
    data = struct("x", d.Xtrain, "y", d.ytrain);
endfunction

function[tree] = train(d, questions)
    data = preprocess_train(d);
    tree = decisiontree(data, splits);
endfunction

function[tree] = decisiontree(node, questions)
    numpoints = size(node.y)(1);
    if numpoints > 5 && length(questions) > 0
        [question, split] = decide(node, questions);
        newquestions = copyquestions(questions);
        newquestions(question, :) = [];
        left = decisiontree(split.left, newquestions);
        right = decisiontree(split.right , newquestions);
        tree = make_node(questions{question}, left, right);
    else
        leaf = make_leaf(node);
        tree = leaf;
    endif
endfunction

function[index, split] = decide(node, questions)
    inf = impurity(node);
    best = -1;
    for i = 1:length(questions)
        [s, gain] = choose(node, questions{i});
        if gain > best
            question = i;
            split = s;
            best = gain;
        endif
    endfor
    index = question;
    value = best;
endfunction

function[split, gain] = choose(node, question)
    num = size(node.x)(1);
    split = question(node);
    inf = impurity(node);
    p = size(split.left.x)(1)/num;
    gain = inf - p*impurity(split.left)-(1-p)*impurity(split.right);
endfunction

function[questions] = copyquestions(qs)
    s = size(qs)(1);
    questions = cell(s,1);
    for i = 1:s
        questions{i} = qs{i};
    endfor
endfunction
