function clearImage() {
    var canvas = $('#sketch')[0];
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    $('#sketch').sketch().actions = [];
}

function postImage() {
    console.log('Submitting for classification!');
    var dataURL = $('#sketch')[0].toDataURL("image/png");
    $.post('/classify',
            {'imageb64': dataURL, 'limit': 5}).done(function(data) {
        console.log(data);
        labels = data['labels'];
        scores = data['scores'];

        drawTable(labels, scores);
        
    });
}

function drawTable(labels, scores) {
    $('#results').empty();
    var table = $('<table></table>');
    var head = $('<tr><th>Label</th><th>Score</th></tr>');
    table.append(head);
    for (i = 0; i < labels.length; ++i) {
        var row = $('<tr></tr>');
        var label = $('<td></td>').text(labels[i]);
        var score = $('<td></td>').text(scores[i]);
        row.append(label);
        row.append(score);
        table.append(row);
    }
    $('#results').append(table);
}

$(function() {
    $('#sketch').sketch({defaultColor: 'white', defaultSize: 20});
    $('#clear-sketch').click(clearImage);
    $('#submit-sketch').click(postImage);
    console.log('Initialized!');
});
