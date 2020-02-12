function newX = reg_fare(X)

    fareX = X(:,3);
    fareX(fareX < 15) = 1;
    fareX(fareX >= 15 & fareX < 30) = 2;
    fareX(fareX >= 30 & fareX < 55) = 3;
    fareX(fareX >= 55) = 4;
    newX = [X(:,1),X(:,2),fareX(:,1),X(:,4),X(:,5),X(:,6),X(:,7), X(:,8)];

end;