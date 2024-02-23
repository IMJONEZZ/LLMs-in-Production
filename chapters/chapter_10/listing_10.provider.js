// Create inline completion provider, this makes suggestions inline
const provider = {
    provideInlineCompletionItems: async (
            document, position, context, token
        ) => {
        // Inline suggestion code goes here

    }
};

// Add provider to Python files
vscode.languages.registerInlineCompletionItemProvider(
    { scheme: 'file', language: 'python' },
    provider
);

// Example of adding provider to all languages
vscode.languages.registerInlineCompletionItemProvider(
    { pattern: '**' },
    provider
);
