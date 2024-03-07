const vscode = require('vscode');

// This method is called when your extension is activated
/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  	// Create inline completion provider, this makes suggestions inline
	const provider = {
		provideInlineCompletionItems: async (
				document, position, context, token
			) => {
			// Grab VSCode editor and selection
			const editor = vscode.window.activeTextEditor;
			const selection = editor.selection;
			const manualKind = 0;
			const manuallyTriggered = context.triggerKind == manualKind;

			// If highlighted back to front, put cursor at the end and rerun
			if (manuallyTriggered && position.isEqual(selection.start)) {
				editor.selection = new vscode.Selection(
					selection.start, selection.end
				);
				vscode.commands.executeCommand(
					"editor.action.inlineSuggest.trigger"
				);
				return []
			}

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
		}
	};

	// Add provider to Python files
	vscode.languages.registerInlineCompletionItemProvider(
		{ scheme: 'file', language: 'python' },
		provider
	);
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
  activate,
  deactivate
}