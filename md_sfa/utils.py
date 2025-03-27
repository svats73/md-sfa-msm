def parse_extra_args(options, extra_args):
    i = 0
    while i < len(extra_args):
        token = extra_args[i]

        # Must start with '--'
        if not token.startswith('--'):
            raise click.ClickException(f"Unknown argument '{token}' (no '--' prefix).")

        # Remove the leading '--'
        token = token[2:]  # e.g. "analysis.safety_percent=0.2"

        # Check if there's an '=' in the same token
        if '=' in token:
            key_part, val_str = token.split('=', 1)
            value = val_str
        else:
            # If no '=', then next token is the value (or True if not present)
            if (i + 1 < len(extra_args)) and (not extra_args[i+1].startswith('--')):
                value = extra_args[i+1]
                i += 1
            else:
                # No explicit next token, so interpret as boolean True
                value = True
            key_part = token

        # Now key_part might look like "analysis.safety_percent"
        # Split on the first dot
        if '.' not in key_part:
            raise click.ClickException(
                f"Extra argument '{key_part}' must include a dot for nested key (e.g. analysis.safety_percent=0.2)."
            )

        top_key, sub_key = key_part.split('.', 1)

        # Ensure top_key is in our defaults
        if top_key not in options:
            raise click.ClickException(f"Unknown top-level key '{top_key}'.")
        if not isinstance(options[top_key], dict):
            raise click.ClickException(f"'{top_key}' in DEFAULT is not a dictionary.")

        # Ensure sub_key is valid in that dict
        if sub_key not in options[top_key]:
            raise click.ClickException(
                f"Sub-key '{sub_key}' not found in DEFAULT['{top_key}']. "
                "If you want to allow brand-new keys, remove this check."
            )

        # Set the value
        options[top_key][sub_key] = value

        i += 1