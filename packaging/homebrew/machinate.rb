class Machinate < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinate"
  url "https://github.com/jmcguigan10/Machinate/releases/download/v0.2.0/machinate-0.2.0.tar.gz"
  sha256 "82ab9350ed1a53b6bd5ff3c4d4209b469401c867183484702651609fa769e38b"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinate", shell_output("#{bin}/macht --help")
  end
end
