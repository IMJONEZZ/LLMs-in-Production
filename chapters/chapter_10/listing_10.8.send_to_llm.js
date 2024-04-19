// On activation send highlighted text to LLM for suggestions
if (manuallyTriggered && selection && !selection.isEmpty) {
	// Grab highlighted text
	const selectionRange = new vscode.Range(
		selection.start, selection.end
	);
	const highlighted = editor.document.getText(
		selectionRange
	);
	
	// Send highlighted text to LLM API
	var payload = {
		prompt: highlighted
	};

	const response = await fetch(
		'http://localhost:8000/generate', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(payload),
	});

	// Return response as suggestion to VSCode editor
	var responseText = await response.text();
	
	range = new vscode.Range(selection.end, selection.end)
	return new Promise(resolve => {
		resolve([{ insertText: responseText, range }])
	})
}
