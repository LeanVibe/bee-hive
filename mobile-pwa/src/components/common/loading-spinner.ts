import { LitElement, html, css } from 'lit'
import { customElement, property } from 'lit/decorators.js'

@customElement('loading-spinner')
export class LoadingSpinner extends LitElement {
  @property() declare size: 'small' | 'medium' | 'large'
  @property() declare color: string
  @property() declare text: string
  
  constructor() {
    super()
    
    // Initialize properties
    this.size = 'medium'
    this.color = '#3b82f6'
    this.text = ''
  }
  
  static styles = css`
    :host {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      gap: 0.75rem;
    }
    
    .spinner {
      border-radius: 50%;
      border-style: solid;
      border-color: transparent;
      border-top-color: var(--spinner-color, #3b82f6);
      animation: spin 1s linear infinite;
    }
    
    .spinner.small {
      width: 16px;
      height: 16px;
      border-width: 2px;
    }
    
    .spinner.medium {
      width: 24px;
      height: 24px;
      border-width: 2px;
    }
    
    .spinner.large {
      width: 40px;
      height: 40px;
      border-width: 3px;
    }
    
    .text {
      font-size: 0.875rem;
      color: #6b7280;
      text-align: center;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `
  
  render() {
    return html`
      <div 
        class="spinner ${this.size}"
        style="--spinner-color: ${this.color}"
      ></div>
      ${this.text ? html`<div class="text">${this.text}</div>` : ''}
    `
  }
}