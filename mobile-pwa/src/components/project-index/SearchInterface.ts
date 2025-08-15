/**
 * Search Interface Component - Placeholder
 * 
 * Project-wide search capabilities with advanced filtering.
 */

import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';

@customElement('search-interface')
export class SearchInterface extends LitElement {
  @property({ type: String }) declare projectId: string;
  @property({ type: Boolean }) declare compact: boolean;

  static styles = css`
    :host {
      display: block;
      padding: 2rem;
      text-align: center;
      background: var(--surface-color, #ffffff);
      border-radius: 0.5rem;
      border: 1px solid var(--border-color, #e5e7eb);
    }
  `;

  render() {
    return html`
      <h3>Search Interface</h3>
      <p>Project-wide search coming soon...</p>
    `;
  }
}